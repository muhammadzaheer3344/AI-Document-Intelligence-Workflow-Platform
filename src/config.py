"""
config.py
---------
Centralised configuration.  Reads from environment variables (populated by
.env via python-dotenv when running locally, or from the host environment on
Streamlit Cloud / any other deployment).

Import this module instead of calling os.getenv() scattered across the app.
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env if present (no-op when the file doesn't exist)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DB_PATH", str(ROOT / "data" / "documents.db")))
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", str(ROOT / "storage")))
MODELS_DIR = ROOT / "models"

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({"pdf", "jpg", "jpeg", "png"})
SUPPORTED_MIME_TYPES: frozenset[str] = frozenset({
    "application/pdf",
    "image/jpeg",
    "image/png",
})

# ---------------------------------------------------------------------------
# RAG / AI assistant
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "300"))
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.55"))
