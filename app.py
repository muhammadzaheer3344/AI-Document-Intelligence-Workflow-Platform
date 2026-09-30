"""
app.py
------
Zyroo AI/ML Internship — Week 3
AI Document Intelligence & Workflow Platform

Streamlit front-end tying together:
  Upload -> Extract text (+OCR fallback) -> Clean -> Classify -> Extract
  fields -> Handle missing fields -> Show result

Two tabs:
  1. Upload & Process  — the actual document intelligence flow.
  2. Model Evaluation   — training-time metrics (accuracy/precision/recall/
     F1, confusion matrix, model comparison) computed by
     models/train_classifier.py, so graders can see Step 6 without
     re-running training themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import date

import streamlit as st

from src.classifier import METADATA_PATH, classify_document
from src.extract_text import extract_text
from src.field_extraction import extract_fields, get_missing_fields
from src.preprocess import clean_text, is_usable, normalize_for_classification
from src.document_repository import DuplicateDocumentError, DocumentRepository
from src.workflow import LOW_CONFIDENCE_THRESHOLD, decide_next_action, process_batch

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "data" / "documents.db"
STORAGE_ROOT = ROOT / "storage"

SUPPORTED_TYPES = ["pdf", "jpg", "jpeg", "png"]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

st.set_page_config(page_title="Zyroo Document Intelligence", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict | None:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return None


@st.cache_resource
def get_repository() -> DocumentRepository:
    return DocumentRepository(DB_PATH, STORAGE_ROOT)


def confidence_badge(confidence: float | None) -> str:
    if confidence is None:
        return ""
    pct = round(confidence * 100)
    if pct >= 80:
        color = "green"
    elif pct >= 55:
        color = "orange"
    else:
        color = "red"
    return f":{color}[Confidence: {pct}%]"


def render_upload_tab() -> None:
    st.subheader("Upload a document")
    st.caption("Supported formats: PDF, JPG, JPEG, PNG. Scanned/photographed documents are "
               "handled automatically via OCR.")

    uploaded_file = st.file_uploader("Choose a file", type=SUPPORTED_TYPES)

    if uploaded_file is None:
        st.info("Upload a document to see extraction, classification, and field results.")
        return

    file_bytes = uploaded_file.read()
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        st.error("This file is larger than the 10 MB upload limit.")
        return

    repository = get_repository()
    existing = repository.get_by_hash(repository.hash_bytes(file_bytes))
    if existing:
        st.info(f"Duplicate detected. This file is already saved as document #{existing['id']}.")
        render_document_detail(existing)
        return

    with st.spinner("Reading document (extracting text, running OCR if needed)…"):
        extraction = extract_text(file_bytes, file_ext)

    if not extraction.success:
        st.error("Could not process this file.")
        for w in extraction.warnings:
            st.warning(w)
        try:
            saved = repository.create_document(file_bytes=file_bytes, original_filename=uploaded_file.name,
                                               document_type="Other", status="New",
                                               workflow_reason="Document text could not be extracted.",
                                               validation_errors={"Readable text": "Document text could not be extracted."})
            repository.transition_document(saved["id"], "Processing", action="Processing started")
            repository.transition_document(saved["id"], "Failed", action="Processing failed",
                                           reason="Document text could not be extracted.")
        except Exception:
            st.error("The failed upload could not be saved to the document repository.")
        return

    for w in extraction.warnings:
        st.warning(w)

    cleaned = clean_text(extraction.text)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Document Info")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Pages:** {extraction.page_count or 1}")
        method_label = {"pymupdf": "Direct PDF text", "ocr": "OCR (image)",
                         "ocr_fallback": "OCR (scanned PDF)"}.get(extraction.method, extraction.method)
        st.write(f"**Extraction method:** {method_label}")
        st.write(f"**OCR used:** {'Yes' if extraction.used_ocr else 'No'}")

    if not is_usable(cleaned):
        st.warning("Very little readable text was found in this document. "
                   "It may be blank, too low-resolution, or not a supported document type.")
        with col_right:
            st.markdown("### Extracted Text")
            st.text_area("Raw text", cleaned or "(empty)", height=150, label_visibility="collapsed")
        try:
            saved = repository.create_document(file_bytes=file_bytes, original_filename=uploaded_file.name,
                                               document_type="Other", status="New", text_preview=cleaned,
                                               workflow_reason="Very little readable text was found.",
                                               validation_errors={"Readable text": "No usable text could be extracted."})
            repository.transition_document(saved["id"], "Processing", action="Processing started")
            saved = repository.transition_document(
                saved["id"], "Needs Review", action="Automated review required",
                reason="Very little readable text was found.",
                validation_errors={"Readable text": "No usable text could be extracted."},
            )
            st.info(f"Saved as document #{saved['id']} with status Needs Review.")
        except Exception:
            st.error("The document could not be saved to the repository.")
        return

    normalized = normalize_for_classification(cleaned)
    classification = classify_document(cleaned, normalized)
    fields = extract_fields(cleaned, classification.label)
    decision = decide_next_action(classification.label, fields, classification.confidence)
    missing = get_missing_fields(fields)
    try:
        saved = repository.create_document(
            file_bytes=file_bytes, original_filename=uploaded_file.name,
            document_type=classification.label, status="New", fields=fields,
            text_preview=cleaned, predicted_type=classification.label,
            confidence=classification.confidence, validation_errors=decision.validation_errors,
            workflow_reason=decision.reason,
        )
        repository.transition_document(saved["id"], "Processing", action="Processing started")
        saved = repository.transition_document(
            saved["id"], decision.action, action="Automated workflow decision",
            reason=decision.reason, validation_errors=decision.validation_errors,
        )
        status = saved["status"]
        st.success(f"Saved as document #{saved['id']} with status {status}.")
    except DuplicateDocumentError as duplicate:
        st.info(f"Duplicate detected. This file is already saved as document #{duplicate.document['id']}.")
    except Exception:
        st.error("The document was processed but could not be saved.")

    with col_left:
        st.markdown("### Classification")
        badge = confidence_badge(classification.confidence)
        if badge:
            st.markdown(f"**Document Type:** {classification.label}  |  {badge}")
        else:
            st.markdown(f"**Document Type:** {classification.label}  "
                        f"*(confidence not available — {classification.method})*")

        st.markdown("### Extracted Fields")
        if fields:
            st.table({"Field": list(fields.keys()), "Value": list(fields.values())})
            if missing:
                st.warning(f"Missing fields ({len(missing)}): {', '.join(missing)}")
            else:
                st.success("All expected fields were found.")
        else:
            st.info("No structured fields are defined for document type "
                    f"'{classification.label}'.")

    with col_right:
        st.markdown("### Extracted Text (cleaned)")
        st.text_area("Cleaned text", cleaned, height=400, label_visibility="collapsed")


def render_evaluation_tab() -> None:
    st.subheader("Model Evaluation")

    metadata = load_model_metadata()
    if metadata is None:
        st.warning("No trained model found yet. Run `python models/train_classifier.py` "
                   "first, then reload this page.")
        return

    st.write(f"**Best model selected:** `{metadata['best_model']}`")
    st.write(f"**Train / test split:** {metadata['train_size']} train / {metadata['test_size']} test")

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
            "F1-score": [f"{v['f1_macro']:.3f}" for v in comparison.values()],
        })

        col1, col2 = st.columns(2)
        cm_path = MODELS_DIR / "confusion_matrix.png"
        cmp_path = MODELS_DIR / "model_comparison.png"
        if cm_path.exists():
            with col1:
                st.image(str(cm_path), caption=f"Confusion matrix — {metadata['best_model']}")
        if cmp_path.exists():
            with col2:
                st.image(str(cmp_path), caption="Macro F1 across candidate models")

        with st.expander("Full per-class classification report (best model)"):
            st.json(report["best_model_full_report"])
    else:
        st.info("Evaluation report not found — re-run training to generate it.")


def render_document_detail(document: dict) -> None:
    st.markdown(f"### Document #{document['id']}: {document['original_filename']}")
    st.write(f"**Type:** {document['document_type']}  |  **Status:** {document['status']}")
    confidence = document.get("confidence")
    confidence_label = f"{confidence:.1%}" if confidence is not None else "Not provided"
    st.write(f"**Predicted type:** {document.get('predicted_type') or document['document_type']}  |  **Confidence:** " + confidence_label)
    if document.get("workflow_reason"):
        st.write(f"**Workflow reason:** {document['workflow_reason']}")
    validation_errors = document.get("validation_errors") or {}
    if validation_errors:
        st.markdown("**Validation results**")
        st.table({"Field": list(validation_errors), "Failure": list(validation_errors.values())})
    elif document.get("document_type") in {"Invoice", "Resume"}:
        st.success("Validation passed.")
    st.write(f"**Uploaded:** {document['upload_date']}  |  **SHA-256:** `{document['file_hash']}`")
    st.write(f"**Stored path:** `{document['file_path']}`")
    fields = document.get("extracted_fields") or {}
    if fields:
        st.table({"Field": list(fields.keys()), "Value": list(fields.values())})
    st.text_area("Text preview", document.get("text_preview") or "(empty)", height=180, disabled=True)
    path = Path(document["file_path"])
    if path.is_file():
        st.download_button("Download document", path.read_bytes(), file_name=document["original_filename"])
    events = get_repository().list_audit_events(document["id"])
    with st.expander("Workflow audit history", expanded=False):
        if events:
            st.dataframe(events, use_container_width=True, hide_index=True)
        else:
            st.info("No workflow history is available.")
    if document["status"] == "Approved" and st.button("Mark workflow complete", key=f"complete_{document['id']}"):
        get_repository().transition_document(document["id"], "Completed", action="Workflow completed")
        st.rerun()


def render_repository_tab() -> None:
    repository = get_repository()
    st.subheader("Document repository")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("Search documents", placeholder="Filename, company, invoice number, type, or text", key="repository_search")
    with col2:
        document_type = st.selectbox("Document type", ["All", "Invoice", "Resume", "Other"], key="repository_type")
    with col3:
        status = st.selectbox("Workflow status", ["All", "New", "Processing", "Needs Review", "Approved", "Rejected", "Completed", "Failed"], key="repository_status")
    date_col1, date_col2, date_col3 = st.columns([1, 1, 1])
    with date_col1:
        start_date = st.date_input("Uploaded from", value=None, key="repository_start_date")
    with date_col2:
        end_date = st.date_input("Uploaded to", value=None, key="repository_end_date")
    with date_col3:
        newest_first = st.radio("Sort", [True, False], format_func=lambda value: "Newest first" if value else "Oldest first", key="repository_sort")
    if st.button("Clear filters"):
        for key, value in {
            "repository_search": "",
            "repository_type": "All",
            "repository_status": "All",
            "repository_start_date": None,
            "repository_end_date": None,
            "repository_sort": True,
        }.items():
            st.session_state[key] = value
        st.rerun()

    documents = repository.list_documents(
        search=search, document_type=document_type, status=status,
        start_date=start_date.isoformat() if isinstance(start_date, date) else None,
        end_date=end_date.isoformat() if isinstance(end_date, date) else None,
        newest_first=newest_first,
    )
    st.caption(f"{len(documents)} saved document(s)")
    if not documents:
        st.info("No documents match the current filters.")
        return
    st.dataframe([
        {"ID": doc["id"], "Filename": doc["original_filename"], "Type": doc["document_type"],
         "Status": doc["status"], "Latest action": doc.get("latest_workflow_action"),
         "Action time": doc.get("latest_workflow_timestamp")}
        for doc in documents
    ], use_container_width=True, hide_index=True)
    options = {f"#{doc['id']} · {doc['original_filename']} · {doc['status']}": doc for doc in documents}
    selected_label = st.selectbox("Select a document", list(options))
    render_document_detail(options[selected_label])


def render_workflow_dashboard() -> None:
    repository = get_repository()
    metrics = repository.get_metrics()
    statuses = metrics["statuses"]
    st.subheader("Workflow metrics")
    metric_columns = st.columns(6)
    for column, label, value in zip(
        metric_columns,
        ("Total documents", "Completed", "Needs review", "Approved", "Rejected", "Failed"),
        (metrics["total"], statuses.get("Completed", 0), statuses.get("Needs Review", 0),
         statuses.get("Approved", 0), statuses.get("Rejected", 0), statuses.get("Failed", 0)),
    ):
        column.metric(label, value)
    st.markdown("#### Documents by type")
    if metrics["by_type"]:
        st.bar_chart(metrics["by_type"])
    else:
        st.info("Document type metrics will appear after the first upload.")
    st.caption(f"Low-confidence review threshold: {LOW_CONFIDENCE_THRESHOLD:.0%}. Confidence is only used when provided by the classifier.")

    st.markdown("#### Batch workflow processing")
    eligible = [doc for doc in repository.list_documents() if doc["status"] in {"New", "Needs Review", "Failed"}]
    choices = {f"#{doc['id']} · {doc['original_filename']} · {doc['status']}": doc for doc in eligible}
    selected = st.multiselect("Select stored documents", list(choices), key="batch_documents")
    if st.button("Run workflow on selected", disabled=not selected):
        results = process_batch(repository, [choices[label]["id"] for label in selected])
        counts = {result: sum(item["Result"] == result for item in results)
                  for result in ("Completed", "Needs Review", "Failed")}
        st.success(
            f"Batch finished: {counts['Completed']} completed, "
            f"{counts['Needs Review']} sent to review, {counts['Failed']} failed."
        )
        st.dataframe(results, use_container_width=True, hide_index=True)


def render_review_queue() -> None:
    repository = get_repository()
    documents = repository.list_documents(status="Needs Review")
    st.subheader("Human review queue")
    st.caption(f"{len(documents)} document(s) require attention")
    if not documents:
        st.success("The review queue is empty.")
        return
    st.dataframe([
        {"ID": doc["id"], "Filename": doc["original_filename"], "Type": doc["document_type"],
         "Status": doc["status"], "Reason": doc.get("workflow_reason") or "Review required"}
        for doc in documents
    ], use_container_width=True, hide_index=True)
    options = {f"#{doc['id']} · {doc['original_filename']}": doc for doc in documents}
    selected = st.selectbox("Document to review", list(options), key="review_document")
    document = options[selected]
    render_document_detail(document)
    reviewer_note = st.text_area("Reviewer note", max_chars=500, key=f"review_note_{document['id']}")
    approve_col, reject_col = st.columns(2)
    if approve_col.button("Approve", key=f"approve_{document['id']}"):
        repository.transition_document(
            document["id"], "Approved", action="Reviewer approved",
            reviewer_note=reviewer_note.strip(),
        )
        st.success("Document approved and audit history updated.")
        st.rerun()
    rejection_reason = reject_col.text_input(
        "Rejection reason (required)", max_chars=500, key=f"reject_reason_{document['id']}"
    )
    if reject_col.button("Reject", key=f"reject_{document['id']}"):
        if not rejection_reason.strip():
            st.error("Enter a short rejection reason before rejecting this document.")
        else:
            repository.transition_document(
                document["id"], "Rejected", action="Reviewer rejected",
                reviewer_note=rejection_reason.strip(),
            )
            st.success("Document rejected and audit history updated.")
            st.rerun()


def main() -> None:
    st.title("📄 Zyroo AI Document Intelligence")
    st.caption("Improved document understanding: cleaner text, more reliable "
               "classification, more useful extraction.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload & Process", "📈 Workflow", "🧑‍⚖️ Review Queue",
        "🗂 Document Repository", "📊 Model Evaluation",
    ])
    with tab1:
        render_upload_tab()
    with tab2:
        render_workflow_dashboard()
    with tab3:
        render_review_queue()
    with tab4:
        render_repository_tab()
    with tab5:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
