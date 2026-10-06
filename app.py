"""
app.py — Zyroo AI Document Intelligence & Workflow Platform (Week 6)
---------------------------------------------------------------------
End-to-end pipeline:
  Upload → Validate → OCR/Extract → Classify → Extract Fields →
  Validate Fields → Anomaly Detection → Workflow Decision →
  Review/Approve/Reject → Audit → RAG/AI Assistant → Search/Report
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path

import streamlit as st

from src.config import (
    DB_PATH, GEMINI_API_KEY, LOW_CONFIDENCE_THRESHOLD,
    MAX_UPLOAD_BYTES, MODELS_DIR, STORAGE_ROOT, SUPPORTED_EXTENSIONS,
)
from src.classifier import METADATA_PATH, classify_document
from src.extract_text import extract_text
from src.field_extraction import extract_fields, get_missing_fields
from src.preprocess import clean_text, is_usable, normalize_for_classification
from src.document_repository import DuplicateDocumentError, DocumentRepository
from src.workflow import decide_next_action, get_workflow_metrics, process_batch, require_human_review
from src.anomaly import detect_anomalies, anomaly_summary
from src.rag import DocumentIndex, answer_question

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Zyroo Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Cached resources (loaded once per Streamlit process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading ML models…")
def get_repository() -> DocumentRepository:
    return DocumentRepository(DB_PATH, STORAGE_ROOT)


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict | None:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return None


@st.cache_resource(show_spinner="Building document index…")
def _build_index_cached(doc_count: int) -> DocumentIndex:  # noqa: ARG001 — count busts cache
    """Rebuild FAISS index when document count changes."""
    repo = get_repository()
    docs = repo.list_documents()
    idx = DocumentIndex()
    idx.build(docs)
    return idx


def get_document_index() -> DocumentIndex:
    repo = get_repository()
    count = len(repo.list_documents())
    return _build_index_cached(count)


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_MAGIC_BYTES = {
    b"%PDF": "pdf",
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG": "png",
}


def _detect_mime(file_bytes: bytes) -> str | None:
    for magic, kind in _MAGIC_BYTES.items():
        if file_bytes[:len(magic)] == magic:
            return kind
    return None


def validate_upload(file_bytes: bytes, filename: str) -> str | None:
    """Return an error string if the upload is invalid, else None."""
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    if len(file_bytes) == 0:
        return "The uploaded file is empty."
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        return f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB size limit."
    detected = _detect_mime(file_bytes)
    if detected is None:
        return "File content does not match a supported format (PDF, JPEG, or PNG)."
    # Map detected kind to allowed extensions
    allowed_for_kind = {"pdf": {"pdf"}, "jpeg": {"jpg", "jpeg"}, "png": {"png"}}
    if ext not in allowed_for_kind.get(detected, {ext}):
        return f"File extension '.{ext}' does not match the actual file content ({detected})."
    return None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def confidence_badge(confidence: float | None) -> str:
    if confidence is None:
        return ""
    pct = round(confidence * 100)
    color = "green" if pct >= 80 else ("orange" if pct >= 55 else "red")
    return f":{color}[{pct}% confidence]"


def render_document_detail(document: dict, *, key_prefix: str) -> None:
    st.markdown(f"**Document #{document['id']}** — {document['original_filename']}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Type:** {document['document_type']}  |  **Status:** {document['status']}")
        conf = document.get("confidence")
        conf_label = f"{conf:.1%}" if conf is not None else "Not available"
        st.write(f"**Predicted type:** {document.get('predicted_type') or document['document_type']}  |  **Confidence:** {conf_label}")
        if document.get("workflow_reason"):
            st.write(f"**Workflow reason:** {document['workflow_reason']}")
    with col2:
        st.write(f"**Uploaded:** {document['upload_date']}")
        st.write(f"**SHA-256:** `{document['file_hash'][:16]}…`")

    validation_errors = document.get("validation_errors") or {}
    if validation_errors:
        st.warning("**Validation failures:**")
        st.table({"Field": list(validation_errors), "Issue": list(validation_errors.values())})
    elif document.get("document_type") in {"Invoice", "Resume"}:
        st.success("Validation passed.")

    fields = document.get("extracted_fields") or {}
    if fields:
        st.markdown("**Extracted fields:**")
        st.table({"Field": list(fields.keys()), "Value": list(fields.values())})

    with st.expander("Text preview"):
        st.text_area(
            "text", document.get("text_preview") or "(empty)", height=160,
            disabled=True, label_visibility="collapsed",
            key=f"{key_prefix}_text_{document['id']}",
        )

    path = Path(document["file_path"])
    if path.is_file():
        st.download_button(
            "⬇ Download document", path.read_bytes(),
            file_name=document["original_filename"],
            key=f"{key_prefix}_dl_{document['id']}",
        )

    events = get_repository().list_audit_events(document["id"])
    st.markdown("**Audit history:**")
    if events:
        st.dataframe(events, use_container_width=True, hide_index=True,
                     key=f"{key_prefix}_audit_{document['id']}")
    else:
        st.info("No audit history available.")

    if document["status"] == "Approved":
        if st.button("Mark Completed", key=f"{key_prefix}_complete_{document['id']}"):
            get_repository().transition_document(
                document["id"], "Completed", action="Workflow completed"
            )
            st.rerun()


# ---------------------------------------------------------------------------
# Tab: Upload & Process
# ---------------------------------------------------------------------------

def render_upload_tab() -> None:
    st.subheader("Upload & Process")
    st.caption(
        "Supported: PDF, JPG, JPEG, PNG · Max 10 MB · "
        "Every saved upload enters the Review Queue for human approval."
    )

    uploaded = st.file_uploader("Choose a file", type=list(SUPPORTED_EXTENSIONS))
    if uploaded is None:
        st.info("Upload a document to begin.")
        return

    file_bytes = uploaded.read()
    t_start = time.perf_counter()

    # --- Security validation ---
    err = validate_upload(file_bytes, uploaded.name)
    if err:
        st.error(err)
        return

    repo = get_repository()
    file_ext = Path(uploaded.name).suffix.lstrip(".").lower()

    # --- Duplicate check ---
    existing = repo.get_by_hash(repo.hash_bytes(file_bytes))
    if existing:
        st.info(f"Duplicate detected — this file is already stored as document #{existing['id']}.")
        render_document_detail(existing, key_prefix="dup")
        return

    # --- Text extraction ---
    with st.spinner("Extracting text (OCR if needed)…"):
        extraction = extract_text(file_bytes, file_ext)

    for w in extraction.warnings:
        st.warning(w)

    if not extraction.success:
        _save_failed_document(repo, file_bytes, uploaded.name, "Document text could not be extracted.")
        return

    cleaned = clean_text(extraction.text)

    # --- Display extraction info ---
    col_left, col_right = st.columns([1, 1])
    method_label = {"pymupdf": "Direct PDF text", "ocr": "OCR (image)",
                    "ocr_fallback": "OCR (scanned PDF)"}.get(extraction.method, extraction.method)
    with col_left:
        st.markdown("#### Document Info")
        st.write(f"**Filename:** {uploaded.name}")
        st.write(f"**Pages:** {extraction.page_count or 1}")
        st.write(f"**Extraction:** {method_label}  |  **OCR used:** {'Yes' if extraction.used_ocr else 'No'}")

    if not is_usable(cleaned):
        st.warning("Very little readable text found — document may be blank or too low-resolution.")
        with col_right:
            st.text_area("Extracted text", cleaned or "(empty)", height=150, disabled=True, label_visibility="collapsed")
        _save_failed_document(repo, file_bytes, uploaded.name, "No usable text could be extracted.")
        return

    # --- Classification ---
    normalized = normalize_for_classification(cleaned)
    classification = classify_document(cleaned, normalized)

    # --- Field extraction ---
    fields = extract_fields(cleaned, classification.label)
    missing = get_missing_fields(fields)

    # --- Anomaly detection ---
    anomalies = detect_anomalies(classification.label, fields, repo)

    # --- Workflow decision ---
    decision = decide_next_action(classification.label, fields, classification.confidence)
    upload_decision = require_human_review(decision)

    # Append anomaly info to workflow reason if errors found
    anomaly_note = anomaly_summary(anomalies)
    if anomaly_note:
        upload_decision = type(upload_decision)(
            upload_decision.action,
            f"{upload_decision.reason} | Anomalies: {anomaly_note}",
            upload_decision.validation_errors,
        )

    # --- Save to repository ---
    try:
        saved = repo.create_document(
            file_bytes=file_bytes, original_filename=uploaded.name,
            document_type=classification.label, status="New", fields=fields,
            text_preview=cleaned, predicted_type=classification.label,
            confidence=classification.confidence,
            validation_errors=upload_decision.validation_errors,
            workflow_reason=upload_decision.reason,
        )
        repo.transition_document(saved["id"], "Processing", action="Processing started")
        saved = repo.transition_document(
            saved["id"], upload_decision.action,
            action="Upload routed to human review",
            reason=upload_decision.reason,
            validation_errors=upload_decision.validation_errors,
        )
        elapsed = time.perf_counter() - t_start
        st.success(f"Saved as document #{saved['id']} · Status: **{saved['status']}** · {elapsed:.1f}s")
    except DuplicateDocumentError as dup:
        st.info(f"Duplicate detected — already stored as document #{dup.document['id']}.")
        return
    except Exception as exc:
        st.error("Document processed but could not be saved to the repository.")
        logging.getLogger(__name__).error("Save failed: %s", exc)
        return

    # --- Results display ---
    with col_left:
        st.markdown("#### Classification")
        badge = confidence_badge(classification.confidence)
        if badge:
            st.markdown(f"**Type:** {classification.label}  |  {badge}")
        else:
            st.markdown(f"**Type:** {classification.label}  *(rule-based, no confidence score)*")

        st.markdown("#### Extracted Fields")
        if fields:
            st.table({"Field": list(fields.keys()), "Value": list(fields.values())})
            if missing:
                st.warning(f"Missing fields: {', '.join(missing)}")
            else:
                st.success("All expected fields found.")
        else:
            st.info(f"No structured fields defined for type '{classification.label}'.")

        # Validation summary
        if upload_decision.validation_errors:
            st.markdown("#### Validation")
            st.warning("Validation failures detected:")
            st.table({
                "Field": list(upload_decision.validation_errors),
                "Issue": list(upload_decision.validation_errors.values()),
            })

        # Anomaly summary
        if anomalies:
            st.markdown("#### Anomaly Detection")
            for flag in anomalies:
                if flag.severity == "error":
                    st.error(f"🚨 {flag.message}")
                else:
                    st.warning(f"⚠️ {flag.message}")
        else:
            st.success("No anomalies detected.")

    with col_right:
        st.markdown("#### Extracted Text")
        st.text_area("cleaned", cleaned, height=420, disabled=True, label_visibility="collapsed")


def _save_failed_document(repo: DocumentRepository, file_bytes: bytes, filename: str, reason: str) -> None:
    try:
        saved = repo.create_document(
            file_bytes=file_bytes, original_filename=filename,
            document_type="Other", status="New",
            workflow_reason=reason,
            validation_errors={"Readable text": reason},
        )
        repo.transition_document(saved["id"], "Processing", action="Processing started")
        repo.transition_document(
            saved["id"], "Needs Review", action="Extraction requires human review",
            reason=reason, validation_errors={"Readable text": reason},
        )
        st.info(f"Saved as document #{saved['id']} with status Needs Review.")
    except DuplicateDocumentError as dup:
        st.info(f"Duplicate — already stored as document #{dup.document['id']}.")
    except Exception as exc:
        st.error("Failed upload could not be saved to the repository.")
        logging.getLogger(__name__).error("Failed save: %s", exc)


# ---------------------------------------------------------------------------
# Tab: AI Assistant (RAG)
# ---------------------------------------------------------------------------

def render_rag_tab() -> None:
    st.subheader("AI Document Assistant")

    repo = get_repository()
    docs = repo.list_documents()
    indexed_docs = [d for d in docs if len((d.get("text_preview") or "").strip()) >= 20]

    if not indexed_docs:
        st.info("No documents with extractable text are stored yet. Upload documents first.")
        return

    st.caption(
        f"{len(indexed_docs)} document(s) indexed · "
        f"{'Gemini AI answers enabled' if GEMINI_API_KEY else 'Retrieval-only mode (set GEMINI_API_KEY for AI answers)'}"
    )

    with st.spinner("Building document index…"):
        try:
            index = get_document_index()
        except RuntimeError as exc:
            st.error(
                f"Could not load the embedding model. {exc}\n\n"
                "Run `pip install tf-keras` to fix a Keras 3 compatibility issue."
            )
            return

    question = st.text_input(
        "Ask a question about your documents",
        placeholder="e.g. What is the total amount on the invoice from Northwind?",
    )

    if not question:
        st.info("Enter a question above to search across all indexed documents.")
        return

    with st.spinner("Searching documents and generating answer…"):
        result = answer_question(question, index)

    if result.error:
        st.warning(f"Note: {result.error}")

    st.markdown("#### Answer")
    st.markdown(result.answer)

    if result.sources:
        with st.expander(f"Sources ({len(result.sources)} retrieved passages)"):
            for i, src in enumerate(result.sources, 1):
                score_pct = round(src["score"] * 100)
                st.markdown(f"**[{i}] {src['filename']}** — relevance: {score_pct}%")
                st.text(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))
                st.divider()

    mode = "Gemini AI" if result.used_llm else "Retrieval-only"
    st.caption(f"Answer mode: {mode}")


# ---------------------------------------------------------------------------
# Tab: Workflow Dashboard
# ---------------------------------------------------------------------------

def render_workflow_dashboard() -> None:
    repo = get_repository()
    metrics = get_workflow_metrics(repo)
    statuses = metrics["statuses"]

    st.subheader("Workflow Dashboard")
    cols = st.columns(7)
    labels_values = [
        ("Total", metrics["total"]),
        ("Completed", statuses.get("Completed", 0)),
        ("Needs Review", statuses.get("Needs Review", 0)),
        ("Approved", statuses.get("Approved", 0)),
        ("Rejected", statuses.get("Rejected", 0)),
        ("Failed", statuses.get("Failed", 0)),
        ("Processing", statuses.get("Processing", 0)),
    ]
    for col, (label, value) in zip(cols, labels_values):
        col.metric(label, value)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Documents by type")
        if metrics["by_type"]:
            st.bar_chart(metrics["by_type"])
        else:
            st.info("No documents yet.")
    with col_b:
        st.markdown("#### Documents by status")
        if statuses:
            st.bar_chart(statuses)
        else:
            st.info("No documents yet.")

    st.caption(
        f"Low-confidence review threshold: {LOW_CONFIDENCE_THRESHOLD:.0%}. "
        "Confidence is only applied when the classifier provides a score."
    )

    st.markdown("#### Batch Workflow Processing")
    eligible = [
        d for d in repo.list_documents()
        if d["status"] in {"New", "Needs Review", "Failed"}
    ]
    choices = {
        f"#{d['id']} · {d['original_filename']} · {d['status']}": d
        for d in eligible
    }
    selected = st.multiselect("Select documents to re-process", list(choices), key="batch_docs")
    if st.button("Run batch workflow", disabled=not selected):
        ids = [choices[label]["id"] for label in selected]
        with st.spinner("Processing batch…"):
            results = process_batch(repo, ids)
        counts = {r: sum(item["Result"] == r for item in results)
                  for r in ("Completed", "Needs Review", "Failed")}
        st.success(
            f"Batch complete: {counts['Completed']} completed, "
            f"{counts['Needs Review']} sent to review, {counts['Failed']} failed."
        )
        st.dataframe(results, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Review Queue
# ---------------------------------------------------------------------------

def render_review_queue() -> None:
    repo = get_repository()
    documents = repo.list_documents(status="Needs Review")
    st.subheader("Human Review Queue")
    st.caption(f"{len(documents)} document(s) awaiting review")

    if not documents:
        st.success("The review queue is empty.")
        return

    st.dataframe([
        {"ID": d["id"], "Filename": d["original_filename"], "Type": d["document_type"],
         "Reason": d.get("workflow_reason") or "Review required"}
        for d in documents
    ], use_container_width=True, hide_index=True)

    options = {f"#{d['id']} · {d['original_filename']}": d for d in documents}
    selected = st.selectbox("Select document to review", list(options), key="review_select")
    document = options[selected]
    render_document_detail(document, key_prefix="review")

    st.markdown("#### Reviewer Action")
    reviewer_note = st.text_area("Reviewer note", max_chars=500, key=f"note_{document['id']}")
    col_approve, col_reject = st.columns(2)

    if col_approve.button("✅ Approve", key=f"approve_{document['id']}"):
        repo.transition_document(
            document["id"], "Approved", action="Reviewer approved",
            reviewer_note=reviewer_note.strip(),
        )
        st.success("Document approved.")
        st.rerun()

    rejection_reason = col_reject.text_input(
        "Rejection reason (required)", max_chars=500, key=f"reject_reason_{document['id']}"
    )
    if col_reject.button("❌ Reject", key=f"reject_{document['id']}"):
        if not rejection_reason.strip():
            st.error("Enter a rejection reason before rejecting.")
        else:
            repo.transition_document(
                document["id"], "Rejected", action="Reviewer rejected",
                reviewer_note=rejection_reason.strip(),
            )
            st.success("Document rejected.")
            st.rerun()


# ---------------------------------------------------------------------------
# Tab: Document Repository
# ---------------------------------------------------------------------------

def render_repository_tab() -> None:
    repo = get_repository()
    st.subheader("Document Repository")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("Search", placeholder="Filename, company, invoice number, type, or text", key="repo_search")
    with col2:
        doc_type = st.selectbox("Type", ["All", "Invoice", "Resume", "Other"], key="repo_type")
    with col3:
        status = st.selectbox("Status", ["All", "New", "Processing", "Needs Review",
                                          "Approved", "Rejected", "Completed", "Failed"], key="repo_status")

    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        start_date = st.date_input("From", value=None, key="repo_start")
    with dcol2:
        end_date = st.date_input("To", value=None, key="repo_end")
    with dcol3:
        newest_first = st.radio("Sort", [True, False],
                                format_func=lambda v: "Newest first" if v else "Oldest first",
                                key="repo_sort")

    if st.button("Clear filters"):
        for k, v in {"repo_search": "", "repo_type": "All", "repo_status": "All",
                     "repo_start": None, "repo_end": None, "repo_sort": True}.items():
            st.session_state[k] = v
        st.rerun()

    documents = repo.list_documents(
        search=search, document_type=doc_type, status=status,
        start_date=start_date.isoformat() if isinstance(start_date, date) else None,
        end_date=end_date.isoformat() if isinstance(end_date, date) else None,
        newest_first=newest_first,
    )
    st.caption(f"{len(documents)} document(s) found")

    if not documents:
        st.info("No documents match the current filters.")
        return

    st.dataframe([
        {"ID": d["id"], "Filename": d["original_filename"], "Type": d["document_type"],
         "Status": d["status"], "Latest action": d.get("latest_workflow_action"),
         "Action time": d.get("latest_workflow_timestamp")}
        for d in documents
    ], use_container_width=True, hide_index=True)

    options = {f"#{d['id']} · {d['original_filename']} · {d['status']}": d for d in documents}
    selected_label = st.selectbox("Inspect document", list(options))
    render_document_detail(options[selected_label], key_prefix="repo")


# ---------------------------------------------------------------------------
# Tab: Model Evaluation
# ---------------------------------------------------------------------------

def render_evaluation_tab() -> None:
    st.subheader("Model Evaluation")
    metadata = load_model_metadata()
    if metadata is None:
        st.warning("No trained model found. Run `python models/train_classifier.py` first.")
        return

    st.write(f"**Best model:** `{metadata['best_model']}`")
    st.write(f"**Train / test split:** {metadata['train_size']} / {metadata['test_size']}")

    eval_path = MODELS_DIR / "evaluation_report.json"
    if eval_path.exists():
        report = json.loads(eval_path.read_text())
        st.markdown("#### Model comparison (macro-averaged)")
        comparison = report["comparison"]
        st.table({
            "Model": list(comparison.keys()),
            "Accuracy": [f"{v['accuracy']:.3f}" for v in comparison.values()],
            "Precision": [f"{v['precision_macro']:.3f}" for v in comparison.values()],
            "Recall": [f"{v['recall_macro']:.3f}" for v in comparison.values()],
            "F1": [f"{v['f1_macro']:.3f}" for v in comparison.values()],
        })
        col1, col2 = st.columns(2)
        cm_path = MODELS_DIR / "confusion_matrix.png"
        cmp_path = MODELS_DIR / "model_comparison.png"
        if cm_path.exists():
            col1.image(str(cm_path), caption=f"Confusion matrix — {metadata['best_model']}")
        if cmp_path.exists():
            col2.image(str(cmp_path), caption="Macro F1 across candidate models")
        with st.expander("Full per-class report"):
            st.json(report["best_model_full_report"])
    else:
        st.info("Evaluation report not found — re-run training to generate it.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📄 Zyroo AI Document Intelligence")
    st.caption(
        "Upload → Extract → Classify → Validate → Anomaly Check → "
        "Workflow → Review → Audit → AI Assistant"
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Upload & Process",
        "🤖 AI Assistant",
        "📈 Workflow",
        "🧑‍⚖️ Review Queue",
        "🗂 Repository",
        "📊 Model Evaluation",
    ])
    with tab1:
        render_upload_tab()
    with tab2:
        render_rag_tab()
    with tab3:
        render_workflow_dashboard()
    with tab4:
        render_review_queue()
    with tab5:
        render_repository_tab()
    with tab6:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
