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

import streamlit as st

from src.classifier import METADATA_PATH, classify_document
from src.extract_text import extract_text
from src.field_extraction import extract_fields, get_missing_fields
from src.preprocess import clean_text, is_usable, normalize_for_classification

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

SUPPORTED_TYPES = ["pdf", "jpg", "jpeg", "png"]

st.set_page_config(page_title="Zyroo Document Intelligence", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict | None:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return None


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

    with st.spinner("Reading document (extracting text, running OCR if needed)…"):
        extraction = extract_text(file_bytes, file_ext)

    if not extraction.success:
        st.error("Could not process this file.")
        for w in extraction.warnings:
            st.warning(w)
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
        return

    normalized = normalize_for_classification(cleaned)
    classification = classify_document(cleaned, normalized)
    fields = extract_fields(cleaned, classification.label)
    missing = get_missing_fields(fields)

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


def main() -> None:
    st.title("📄 Zyroo AI Document Intelligence")
    st.caption("Improved document understanding: cleaner text, more reliable "
               "classification, more useful extraction.")

    tab1, tab2 = st.tabs(["📤 Upload & Process", "📊 Model Evaluation"])
    with tab1:
        render_upload_tab()
    with tab2:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
