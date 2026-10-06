# AI Document Intelligence & Workflow Platform

An end-to-end document processing platform built with Streamlit.  
Upload a PDF or image — the app extracts text (with OCR fallback for scanned documents), classifies the document, extracts structured fields, validates them, checks for anomalies, routes through a workflow engine, and supports human review, approval, rejection, and AI-assisted querying.

🚀 **Live Demo:** [Open on Streamlit Cloud](https://ai-document-intelligence-workflow-platform-pmmcfn6ruedfri5gpp9.streamlit.app)

---

## Features

- **PDF & image support** — native PDF text extraction via PyMuPDF; automatic OCR fallback (Tesseract) for scanned PDFs and image files (JPG, PNG)
- **Image preprocessing** — grayscale, upscaling, denoising, and adaptive thresholding before OCR
- **Document classification** — TF-IDF + Logistic Regression ML model with rule-based fallback; confidence scores shown
- **Structured field extraction** — regex/keyword extraction per document type with "Not Found" handling
- **Field validation** — required-field checks, email/phone/date/amount format validation
- **Anomaly detection** — duplicate hash detection, repeated invoice number detection, financial sanity checks (zero, negative, unusually large amounts)
- **RAG / AI assistant** — FAISS vector index over stored document text, sentence-transformer embeddings, Gemini-powered grounded answers (retrieval-only fallback when no API key is set)
- **Workflow engine** — `New → Processing → Needs Review → Approved/Rejected → Completed/Failed` with enforced transitions
- **Human review queue** — reviewers inspect extracted fields, validation failures, anomalies, and approve or reject with a required reason
- **Audit history** — every upload, transition, and reviewer action is timestamped per document
- **Batch processing** — re-evaluate multiple stored documents; one failure does not stop the batch
- **Workflow dashboard** — counts by status and type, bar charts, batch processing UI
- **Document repository** — search, filter by type/status/date, sort, inspect, download
- **Model evaluation tab** — accuracy, precision, recall, F1, confusion matrix, model comparison chart
- **Security** — file type + MIME validation, size limits, safe filenames, no raw tracebacks shown to users
- **Configuration** — all secrets and tunable settings via environment variables / `.env`

---

## Architecture

```
Upload
  ↓ File type / MIME / size / integrity validation
OCR / Text Extraction  (PyMuPDF → Tesseract fallback)
  ↓
Text Cleaning / Normalization  (preprocess.py)
  ↓
Classification  (TF-IDF + Logistic Regression, rule-based fallback)
  ↓
Structured Field Extraction  (regex per document type)
  ↓
Field Validation  (required fields, format checks)
  ↓
Anomaly Detection  (duplicate hash, repeated invoice number, financial checks)
  ↓
Workflow Decision  (validation errors + confidence → Needs Review or Completed)
  ↓ (all uploads held for human review)
SQLite Repository  (metadata, fields, audit events, file storage)
  ↓
┌──────────────────┬──────────────────┐
↓                  ↓                  ↓
Approve         Needs Review        Reject
↓                  ↓                  ↓
Completed       Reviewer UI        Rejected
                   ↓
               Audit History

RAG / AI Assistant  (FAISS index → sentence-transformer embeddings → Gemini or retrieval-only)
Search / Reporting  (SQLite full-text search, filters, dashboard metrics)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Tesseract must be installed at the OS level:

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows — https://github.com/UB-Mannheim/tesseract/wiki
```

> **Note:** If you see a `Keras 3` error when using the AI Assistant, run:
> ```bash
> pip install tf-keras
> ```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Optional | Google Gemini API key for AI-generated answers. Without it, the assistant returns retrieved passages directly. Get a free key at https://aistudio.google.com |
| `GEMINI_MODEL` | Optional | Gemini model name (default: `gemini-1.5-flash`) |
| `EMBED_MODEL_NAME` | Optional | Sentence-transformer model (default: `all-MiniLM-L6-v2`) |
| `RAG_TOP_K` | Optional | Number of passages to retrieve (default: `5`) |
| `RAG_CHUNK_SIZE` | Optional | Words per chunk (default: `300`) |
| `RAG_CHUNK_OVERLAP` | Optional | Overlap between chunks (default: `50`) |
| `LOW_CONFIDENCE_THRESHOLD` | Optional | Confidence below which documents go to review (default: `0.55`) |
| `MAX_UPLOAD_BYTES` | Optional | Upload size limit in bytes (default: `10485760` = 10 MB) |
| `DB_PATH` | Optional | Override SQLite database path |
| `STORAGE_ROOT` | Optional | Override file storage directory |

**Never commit `.env` to version control.** It is listed in `.gitignore`.

---

## Running the Application

```bash
# 1. Generate training dataset
python data/generate_dataset.py

# 2. Train and evaluate the classifier
python models/train_classifier.py

# 3. (Optional) regenerate test fixtures
python tests/generate_sample_docs.py

# 4. Launch the app
streamlit run app.py

# 5. Run all tests
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Supported Documents

| Type | Description |
|---|---|
| Invoice | PDF or image invoices with invoice number, date, company, and total amount |
| Resume | PDF or image CVs/resumes with name, email, phone, and skills |
| Other | Any other document — classified, stored, and routed for review |

Supported file formats: **PDF, JPG, JPEG, PNG** (max 10 MB).

---

## Extraction Fields

| Document Type | Fields |
|---|---|
| Invoice | Invoice Number, Date, Company Name, Total Amount |
| Resume | Name, Email, Phone, Skills |

Fields that cannot be found are stored as `"Not Found"` and flagged as missing.

---

## Workflow Rules

States: `New → Processing → Needs Review → Approved → Completed`  
Alternative paths: `→ Rejected`, `→ Failed → Processing` (retry)

**Routing logic:**
- All new uploads are held in `Needs Review` for human review, even when automated checks pass.
- Validation failures (missing required fields, invalid formats) → `Needs Review`
- Classification confidence below 55% (when provided) → `Needs Review`
- Extraction failures → `Needs Review`
- Anomalies (duplicate invoice number, zero/negative amount) → noted in workflow reason

**Batch processing** re-runs validation and routing on stored documents without repeating OCR or classification.

---

## Review Process

The **Review Queue** tab shows all documents in `Needs Review` status.

Reviewers can:
1. Inspect extracted fields, validation failures, anomaly flags, and audit history
2. Add a reviewer note
3. **Approve** — moves document to `Approved` (then manually to `Completed`)
4. **Reject** — requires a rejection reason; moves document to `Rejected`

Every action is recorded in the audit history with a timestamp.

---

## AI Assistant

The **AI Assistant** tab provides document-grounded question answering:

1. **Indexing** — all stored documents with extractable text are chunked into overlapping word windows and embedded using `all-MiniLM-L6-v2` (a local sentence-transformer model, no API key needed).
2. **Retrieval** — the question is embedded and the top-k most similar passages are retrieved via FAISS cosine similarity.
3. **Answer generation** — if `GEMINI_API_KEY` is set, Gemini generates an answer grounded in the retrieved passages. Otherwise, the most relevant passage is returned directly.
4. **Source references** — retrieved passages are shown with their source document filename and relevance score.
5. **Unanswerable questions** — if the top passage has cosine similarity below 0.15, the assistant states that the available documents do not contain enough information.

**Limitations:**
- The assistant only knows what is in the stored `text_preview` field (up to 2000 characters per document). Full-document indexing would require storing complete extracted text.
- Without a Gemini API key, answers are retrieval-only (no synthesis across multiple passages).
- The FAISS index is rebuilt per session; for large document sets this adds startup latency.

---

## Testing

Run the full suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Test Results (Week 6)

| # | Test | Result |
|---|---|---|
| 1 | Clean digital PDF — full pipeline | ✅ Pass |
| 2 | Scanned document — OCR path | ✅ Pass |
| 3 | Invoice and Resume classification | ✅ Pass |
| 4 | Missing required fields → Needs Review | ✅ Pass |
| 5 | Exact duplicate raises; repeated invoice number flagged | ✅ Pass |
| 6 | Unsupported file type (.docx) rejected; empty file rejected | ✅ Pass |
| 7 | Corrupt PDF handled without crash; blank image returns no usable text | ✅ Pass |
| 8 | Low confidence → Needs Review; absent confidence not invented | ✅ Pass |
| 9 | Zero/negative/large amounts flagged; normal amount clean | ✅ Pass |
| 10 | Invalid transitions rejected; valid full path succeeds | ✅ Pass |
| 11 | Mixed-success batch: one failure does not stop others | ✅ Pass |
| 12 | Records survive repository reconnect (restart simulation) | ✅ Pass |
| 13 | RAG: relevant passage retrieved; source document referenced | ⏭ Skip* |
| 14 | RAG: unanswerable question returns clear message; empty index handled | ✅ Pass (empty index); ⏭ Skip* (embedding) |

*Tests 13 and the embedding-dependent part of 14 are skipped on this machine due to a `Keras 3` / `sentence-transformers` version conflict. Fix with `pip install tf-keras`. The RAG module itself is fully implemented and functional in the Streamlit app once the dependency is resolved.

**Total: 46 tests run, 43 pass, 3 skip, 0 fail.**

Week 5 suite: 18 tests, all passing.  
Week 6 suite: 28 new tests, 25 pass, 3 skip (environment-only issue).

---

## Evaluation

Best model: **Logistic Regression** (TF-IDF features)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 |
| Linear SVM | 1.000 | 1.000 | 1.000 | 1.000 |
| Naive Bayes | 1.000 | 1.000 | 1.000 | 1.000 |

*Note: Perfect scores reflect the synthetic training dataset. Real-world performance on diverse documents will be lower.*

---

## Known Limitations

- Text preview stored in the database is capped at 2000 characters; the RAG assistant only sees this truncated text.
- OCR quality depends on Tesseract and image resolution; very low-quality scans may produce poor extraction.
- The ML classifier was trained on synthetic data; it may misclassify unusual real-world documents.
- The RAG assistant requires `tf-keras` on systems with Keras 3 installed alongside `sentence-transformers`.
- No user authentication — all users share the same document repository.
- Streamlit Cloud deployment requires Tesseract via `packages.txt` (already configured).

---

## Future Improvements

- Store full extracted text (not just 2000-char preview) for better RAG coverage
- Add user authentication and per-user document isolation
- Support additional document types (contracts, purchase orders, receipts)
- Add line-item extraction for invoices to enable full financial reconciliation
- Retrain the classifier on real-world document samples
- Add a persistent FAISS index file to avoid rebuilding on every session

---

## Deployment

### Streamlit Cloud

1. Push the repository to GitHub (`.env` and `data/documents.db` are gitignored).
2. Connect the repository to [Streamlit Cloud](https://streamlit.io/cloud).
3. Set `GEMINI_API_KEY` (and any other variables) in the Streamlit Cloud **Secrets** panel.
4. `packages.txt` installs Tesseract and required system libraries automatically.
5. The app starts with `streamlit run app.py`.

### Local

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY if desired
python data/generate_dataset.py
python models/train_classifier.py
streamlit run app.py
```

---

## Project Structure

```
AI-Document-Intelligence-Workflow-Platform/
├── app.py                          # Streamlit entry point (6 tabs)
├── requirements.txt
├── packages.txt                    # System apt dependencies for Streamlit Cloud
├── .env.example                    # Environment variable template (no secrets)
├── src/
│   ├── config.py                   # Centralised configuration (reads .env)
│   ├── extract_text.py             # PDF + OCR text extraction
│   ├── preprocess.py               # Text cleaning / normalization
│   ├── classifier.py               # ML + rule-based classification
│   ├── field_extraction.py         # Regex field extraction
│   ├── validator.py                # Required-field and format validation
│   ├── anomaly.py                  # Duplicate / financial anomaly detection
│   ├── workflow.py                 # Workflow rules and batch processing
│   ├── document_repository.py      # SQLite CRUD, transitions, audit, file storage
│   └── rag.py                      # Chunking, FAISS embeddings, RAG answer generation
├── data/
│   ├── generate_dataset.py
│   └── dataset.csv
├── models/
│   ├── train_classifier.py
│   ├── best_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── model_metadata.json
│   ├── evaluation_report.json
│   ├── confusion_matrix.png
│   └── model_comparison.png
├── tests/
│   ├── generate_sample_docs.py
│   ├── sanity_check.py
│   ├── ocr_test.py
│   ├── test_document_repository.py  # Week 5: 5 tests
│   ├── test_workflow.py             # Week 5: 13 tests
│   └── test_week6_integration.py    # Week 6: 28 tests
└── sample_docs/                     # Test fixtures
```
