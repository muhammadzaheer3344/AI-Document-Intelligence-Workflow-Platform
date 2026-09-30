# AI Document Intelligence & Workflow Platform

An end-to-end document processing platform built with Streamlit. Upload a PDF or image — the app extracts text (with OCR fallback for scanned documents), classifies the document as an **Invoice**, **Resume**, or **Other**, and pulls out the relevant structured fields automatically.

🚀 **Live Demo:** [Open on Streamlit Cloud](https://ai-document-intelligence-workflow-platform-pmmcfn6ruedfri5gpp9.streamlit.app)

---

## Features

- **PDF & image support** — native PDF text extraction via PyMuPDF; automatic OCR fallback (Tesseract) for scanned PDFs and image files (JPG, PNG)
- **Image preprocessing** — grayscale conversion, upscaling, denoising, and adaptive thresholding before OCR to improve accuracy on low-quality scans
- **Document classification** — TF-IDF + Logistic Regression ML model (compared against Linear SVM and Naive Bayes); rule-based keyword classifier as fallback
- **Field extraction** — structured fields per document type with "Not Found" handling and missing-field reporting
- **Confidence scores** — shown when the ML model supports `predict_proba`
- **Model evaluation tab** — accuracy, precision, recall, F1, confusion matrix, and model comparison chart rendered in-app
- **Persistent document repository** — SQLite metadata, SHA-256 duplicate detection, organized UUID-based file storage, search, filters, sorting, and detail/download view
- **Workflow state management** — SQLite-backed `New`, `Processing`, `Needs Review`, `Approved`, `Rejected`, `Completed`, and `Failed` states with enforced transitions
- **Field validation and automated decisions** — required invoice/resume fields and email, phone, date, and amount checks route invalid data for review
- **Confidence-aware review** — classifier confidence is saved and the documented 55% threshold only applies when the model supplies confidence
- **Human review queue** — reviewers inspect extracted fields and validation failures, approve documents, or reject with a required reason
- **Audit history** — uploads, automated decisions, reviewer actions, batch outcomes, and migration events are timestamped per document
- **Batch workflow processing** — re-evaluate multiple stored documents independently and see completed, review, and failed results
- **Workflow dashboard** — counts by state and document type, searchable workflow status, latest action, and action timestamp
- **Upload safety** — only PDF/JPG/JPEG/PNG files up to 10 MB are accepted; unreadable files are handled without exposing raw exceptions

### Extracted Fields

| Document Type | Fields Extracted |
|---|---|
| Invoice | Invoice Number, Date, Company Name, Total Amount |
| Resume | Name, Email, Phone, Skills |

---

## Project Structure

```
AI-Document-Intelligence-Workflow-Platform/
├── app.py                        # Streamlit app (entry point)
├── requirements.txt
├── packages.txt                  # System-level apt dependencies for Streamlit Cloud
├── src/
│   ├── extract_text.py           # PDF text extraction + OCR fallback + image preprocessing
│   ├── preprocess.py             # Text cleaning / normalization
│   ├── classifier.py             # Rule-based baseline + ML model wrapper (inference time)
│   ├── field_extraction.py       # Regex/keyword field extraction, "Not Found" handling
│   ├── validator.py              # Required-field and format validation
│   ├── workflow.py               # Workflow rules, confidence threshold, batch coordinator
│   └── document_repository.py    # SQLite CRUD, transitions, audit, metrics, file storage
├── data/
│   ├── generate_dataset.py       # Builds a balanced synthetic Invoice/Resume/Other dataset
│   └── dataset.csv               # Training data (text, label)
├── models/
│   ├── train_classifier.py       # Trains + compares LogReg/SVM/NaiveBayes, saves best model
│   ├── best_model.joblib         # Trained classifier
│   ├── tfidf_vectorizer.joblib   # Fitted TF-IDF vectorizer
│   ├── model_metadata.json       # Best model name, labels, split sizes
│   ├── evaluation_report.json    # Full accuracy/precision/recall/F1 + confusion matrix
│   ├── confusion_matrix.png      # Confusion matrix plot
│   └── model_comparison.png      # Bar chart comparing candidate models
├── tests/
│   ├── generate_sample_docs.py   # Creates native PDFs, scanned images, blank/corrupt files
│   ├── sanity_check.py           # End-to-end pipeline check on a native PDF
│   ├── ocr_test.py               # End-to-end pipeline check on a scanned image
│   ├── test_document_repository.py # Persistence, transition, migration, and audit tests
│   └── test_workflow.py           # Validation and 12-scenario workflow matrix
└── sample_docs/                  # Test fixtures (native PDFs, scans, blank, corrupt)

data/documents.db                 # Created automatically on first app run (ignored by git)
storage/{invoice,resume,other}/   # UUID-named uploaded files (ignored by git)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Tesseract must be installed at the OS level (`pytesseract` is just a Python wrapper):

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows
# https://github.com/UB-Mannheim/tesseract/wiki
```

---

## Run Locally

```bash
# 1. Generate the training dataset
python data/generate_dataset.py

# 2. Train and evaluate the classifier (writes everything under models/)
python models/train_classifier.py

# 3. (Optional) regenerate test fixtures
python tests/generate_sample_docs.py

# 4. Launch the app
streamlit run app.py

# 5. Run workflow and repository tests
python -m unittest discover -s tests -p "test_*.py"
```

The app has five tabs:
- **Upload & Process** — upload, classify, extract, validate, and automatically route documents
- **Workflow** — metrics and isolated batch processing of stored documents
- **Review Queue** — inspect validation results and approve or reject with a reason
- **Document Repository** — search filename, company, invoice number, type, and text; filter by workflow status/date; sort; inspect audit history; download files
- **Model Evaluation** — accuracy/precision/recall/F1, confusion matrix, and model comparison chart

## Week 5 Workflow Details

New uploads pass through `New` → `Processing` → `Completed`, `Needs Review`, or `Failed`. Reviewers can move `Needs Review` documents to `Approved` or `Rejected`; approved documents can then be marked `Completed`. Invalid transitions are rejected by the repository API. Failed documents may be retried through the batch workflow.

The validator requires Invoice Number, Date, Company Name, and Total Amount for invoices, and Name, Email, and Skills for resumes. It also validates present email, phone, date, and amount values. Every failure is stored by field name. Classifications below 55% confidence are sent for review only when the classifier actually provides a confidence score; rule-based classifications do not receive an invented score.

The `documents` table stores predicted type, optional confidence, validation failures, and the current workflow reason in addition to Week 4 metadata. `audit_events` records action, previous/new status, timestamp, and reason. On startup, existing Week 4 `Processed` rows are migrated to `Completed` and receive a migration history event; their earlier history cannot be reconstructed.

Batch processing uses already stored extracted fields and confidence, so it re-runs validation and routing rather than repeating PDF/OCR extraction or model classification. Each selected document is isolated: a failure is recorded and does not stop later documents in the batch. The repository search supports filename, company, invoice number, document type, and text preview; the workflow table also displays the latest audit action and timestamp.

---

## Testing Results

Full pipeline tested against all fixtures in `sample_docs/`:

| File | Extraction Method | Classified As | Missing Fields |
|---|---|---|---|
| native_invoice.pdf | Direct PDF text | Invoice ✅ | none |
| native_resume.pdf | Direct PDF text | Resume ✅ | none |
| scanned_invoice.png | OCR | Invoice ✅ | none |
| noisy_scan.png (rotated + noise) | OCR | Invoice ✅ | none |
| blank.png | OCR attempted | — | flagged as "no usable text", no crash |
| corrupt.pdf | — | — | flagged as unreadable, no crash |

The workflow unit suite also exercises a 12-case decision matrix: native and OCR-style invoices, multiple amount/date formats, missing invoice fields, invalid date/amount, complete resumes with and without confidence, invalid email, missing skills, low confidence, and an unreadable scan. Repository tests cover duplicate detection, legal and illegal state changes, reviewer audit notes, migration from the Week 4 `Processed` status, mixed-success batch continuation, and independent database-backed search/update/delete behavior.

Run the automated suite with `python -m unittest discover -s tests -p "test_*.py"`. The Week 5 suite currently contains 14 passing tests. The UI should also be exercised locally for approve/reject and batch interactions before submission; screenshots are not committed in this repository.

---

## Completion Checklist

- [x] Cleaned and normalized extracted text
- [x] OCR tested with scanned and noisy/rotated image documents
- [x] ML classifier trained and evaluated (Logistic Regression, Linear SVM, Naive Bayes)
- [x] Improved field extraction with robust regex patterns
- [x] Missing fields handled — every field resolves to a value or "Not Found"
- [x] Confidence scores shown where available
- [x] Full evaluation metrics rendered in-app (accuracy, precision, recall, F1, confusion matrix)
- [x] Workflow states, validated transitions, and migration of Week 4 records
- [x] Required field and format validation with exact failed-field records
- [x] Separate workflow rule engine with optional confidence threshold
- [x] Human review, rejection reason requirement, and audit history
- [x] Batch workflow processing with per-document outcomes
- [x] Workflow metrics, status filtering, search, and latest audit action
- [x] 12-scenario workflow decision test matrix
- [x] App deployed on Streamlit Cloud
