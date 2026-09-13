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
│   └── field_extraction.py       # Regex/keyword field extraction, "Not Found" handling
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
│   └── ocr_test.py               # End-to-end pipeline check on a scanned image
└── sample_docs/                  # Test fixtures (native PDFs, scans, blank, corrupt)
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
```

The app has two tabs:
- **Upload & Process** — upload a document and see extraction, classification, and field results
- **Model Evaluation** — accuracy/precision/recall/F1, confusion matrix, and model comparison chart

---

## Testing Results

Full pipeline tested against all fixtures in `sample_docs/`:

| File | Extraction Method | Classified As | Missing Fields |
|---|---|---|---|
| native_invoice.pdf | Direct PDF text | Invoice ✅ | none |
| native_resume.pdf | Direct PDF text | Resume ✅ | none |
| scanned_invoice.png | OCR | Invoice ✅ | Invoice Number (OCR misread — expected limitation) |
| noisy_scan.png (rotated + noise) | OCR | Invoice ✅ | none |
| blank.png | OCR attempted | — | flagged as "no usable text", no crash |
| corrupt.pdf | — | — | flagged as unreadable, no crash |

---

## Completion Checklist

- [x] Cleaned and normalized extracted text
- [x] OCR tested with scanned and noisy/rotated image documents
- [x] ML classifier trained and evaluated (Logistic Regression, Linear SVM, Naive Bayes)
- [x] Improved field extraction with robust regex patterns
- [x] Missing fields handled — every field resolves to a value or "Not Found"
- [x] Confidence scores shown where available
- [x] Full evaluation metrics rendered in-app (accuracy, precision, recall, F1, confusion matrix)
- [x] App deployed on Streamlit Cloud
