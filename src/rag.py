"""
rag.py
------
Retrieval-Augmented Generation for the AI Document Intelligence Platform.

Pipeline:
  1. Chunk stored document text into overlapping passages.
  2. Embed passages with a local sentence-transformer model (no API key needed).
  3. Store embeddings in a per-session FAISS index keyed by document ID.
  4. At query time: embed the question, retrieve top-k passages, call the
     configured LLM (Gemini if GEMINI_API_KEY is set, else a local fallback
     that answers directly from retrieved context without an LLM).
  5. Return the answer + source references so the UI can show grounding.

Design decisions:
  - The FAISS index is rebuilt from the DB on each app session (documents are
    small; this keeps the architecture simple and avoids stale index files).
  - The sentence-transformer model is cached via @st.cache_resource so it is
    loaded once per Streamlit process.
  - If no documents are indexed, the assistant says so clearly.
  - If the LLM is unavailable, the assistant returns the raw retrieved passages
    so the user still gets grounded information.
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read from environment; safe defaults for local dev)
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "300"))   # words per chunk
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))  # word overlap


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, doc_id: int, filename: str) -> list[dict[str, Any]]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    chunks: list[dict[str, Any]] = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for start in range(0, len(words), step):
        passage = " ".join(words[start: start + CHUNK_SIZE])
        if passage.strip():
            chunks.append({"doc_id": doc_id, "filename": filename, "text": passage})
    return chunks


# ---------------------------------------------------------------------------
# Embedding model (lazy-loaded; caller should use st.cache_resource wrapper)
# ---------------------------------------------------------------------------

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load embedding model '{EMBED_MODEL_NAME}'. "
                f"If you see a Keras error, run: pip install tf-keras\n"
                f"Original error: {exc}"
            ) from exc
    return _embed_model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embed_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# FAISS index wrapper
# ---------------------------------------------------------------------------

@dataclass
class DocumentIndex:
    chunks: list[dict[str, Any]] = field(default_factory=list)
    _index: Any = field(default=None, repr=False)

    def build(self, documents: list[dict[str, Any]]) -> int:
        """Index all documents that have usable text. Returns chunk count."""
        import faiss  # type: ignore

        all_chunks: list[dict[str, Any]] = []
        for doc in documents:
            text = (doc.get("text_preview") or "").strip()
            if len(text) < 20:
                continue
            all_chunks.extend(chunk_text(text, doc["id"], doc["original_filename"]))

        if not all_chunks:
            self.chunks = []
            self._index = None
            return 0

        texts = [c["text"] for c in all_chunks]
        vectors = embed_texts(texts).astype("float32")
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner-product on L2-normalised = cosine
        index.add(vectors)

        self.chunks = all_chunks
        self._index = index
        return len(all_chunks)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        """Return top-k chunks most relevant to query."""
        if self._index is None or not self.chunks:
            return []
        import faiss  # type: ignore

        q_vec = embed_texts([query]).astype("float32")
        faiss.normalize_L2(q_vec)
        scores, indices = self._index.search(q_vec, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results


# ---------------------------------------------------------------------------
# LLM answer generation
# ---------------------------------------------------------------------------

def _build_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    context_parts = []
    for i, p in enumerate(passages, 1):
        context_parts.append(f"[{i}] (from {p['filename']}):\n{p['text']}")
    context = "\n\n".join(context_parts)
    return textwrap.dedent(f"""
        You are a document analysis assistant. Answer the question using ONLY the
        document excerpts provided below. If the excerpts do not contain enough
        information to answer, say exactly:
        "The available documents do not contain enough information to answer this question."

        Do not invent facts. Cite the excerpt number(s) you used, e.g. [1], [2].

        --- Document Excerpts ---
        {context}
        --- End of Excerpts ---

        Question: {question}
        Answer:
    """).strip()


def _answer_with_gemini(prompt: str) -> str:
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()


def _answer_local_fallback(question: str, passages: list[dict[str, Any]]) -> str:
    """No LLM available: return the most relevant passage as the answer."""
    if not passages:
        return "The available documents do not contain enough information to answer this question."
    best = passages[0]
    snippet = best["text"][:600]
    return (
        f"[Retrieved from {best['filename']}]\n\n{snippet}\n\n"
        f"*(No LLM configured — showing the most relevant document excerpt. "
        f"Set GEMINI_API_KEY in your .env file to enable AI-generated answers.)*"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict[str, Any]]   # list of {doc_id, filename, score, text}
    used_llm: bool
    error: str = ""


def answer_question(
    question: str,
    index: DocumentIndex,
    *,
    top_k: int = TOP_K,
) -> RAGAnswer:
    """Retrieve relevant passages and generate a grounded answer."""
    if not question.strip():
        return RAGAnswer("Please enter a question.", [], False)

    passages = index.search(question, top_k=top_k)
    if not passages:
        return RAGAnswer(
            "No documents have been indexed yet. Upload and process documents first.",
            [], False,
        )

    # Check if top passage is relevant enough (cosine > 0.15)
    if passages[0]["score"] < 0.15:
        return RAGAnswer(
            "The available documents do not contain enough information to answer this question.",
            passages, False,
        )

    prompt = _build_prompt(question, passages)

    if GEMINI_API_KEY:
        try:
            answer = _answer_with_gemini(prompt)
            return RAGAnswer(answer, passages, used_llm=True)
        except Exception as exc:
            logger.warning("Gemini call failed: %s", exc)
            return RAGAnswer(
                _answer_local_fallback(question, passages),
                passages, used_llm=False,
                error=f"LLM unavailable ({type(exc).__name__}): {exc}",
            )

    return RAGAnswer(_answer_local_fallback(question, passages), passages, used_llm=False)
