# Zyroo AI/ML Internship — Week 3
## AI Document Intelligence & Workflow Platform — Task 02: Improve Document Understanding

Builds on the Week 2 MVP: same upload -> classify -> extract -> display flow, but with
cleaner text, a trained ML classifier (compared against a rule-based baseline),
better OCR handling, more robust field extraction, and honest missing-field / confidence
reporting.

## Project Structure

```
zyroo_week3/
├── app.py                        # Streamlit app (entry point)
├── src/
│   ├── extract_text.py           # PDF text extraction + OCR fallback + image preprocessing
│   ├── preprocess.py              # Text cleaning / normalization
│   ├── classifier.py              # Rule-based baseline + ML model wrapper (inference time)
│   └── field_extraction.py        # Regex/keyword field extraction, "Not Found" handling
├── data/
│   ├── generate_dataset.py        # Builds a balanced synthetic Invoice/Resume/Other dataset
│   └── dataset.csv                # Generated training data (text, label)
├── models/
│   ├── train_classifier.py        # Trains + compares LogReg/SVM/NaiveBayes, saves best model
│   ├── best_model.joblib          # Trained classifier (generated)
│   ├── tfidf_vectorizer.joblib     # Fitted TF-IDF vectorizer (generated)
│   ├── model_metadata.json        # Which model won, labels, split sizes (generated)
│   ├── evaluation_report.json     # Full accuracy/precision/recall/F1 + confusion matrix (generated)
│   ├── confusion_matrix.png        # Confusion matrix plot (generated)
│   └── model_comparison.png       # Bar chart comparing candidate models (generated)
├── tests/
│   └── generate_sample_docs.py    # Creates native PDFs, scanned images, blank/corrupt files for testing
├── sample_docs/                   # Generated test fixtures (see above)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt

# Tesseract must be installed at the OS level (pytesseract is just a wrapper)
# Ubuntu/Debian: sudo apt install tesseract-ocr
# macOS:         brew install tesseract
# Windows:       https://github.com/UB-Mannheim/tesseract/wiki
```

## Run it

```bash
# 1. Generate the training dataset
python data/generate_dataset.py

# 2. Train and evaluate the classifier (writes everything under models/)
python models/train_classifier.py

# 3. (Optional) generate test fixtures — native PDFs, scanned images, blank/corrupt files
python tests/generate_sample_docs.py

# 4. Launch the app
streamlit run app.py
```

The app has two tabs: **Upload & Process** (the actual pipeline) and **Model Evaluation**
(accuracy/precision/recall/F1 + confusion matrix + model comparison chart from step 2).

## ⚠️ Important — read before submitting

`data/generate_dataset.py` builds a **synthetic** dataset from templates so the whole
pipeline has something to train/evaluate on immediately. On this synthetic data the
classifier scores ~100% — that's the templates being too easy to tell apart, not a sign
the model is production-ready. **Before submitting, replace or supplement `data/dataset.csv`
with real invoices/resumes/other documents you collect** (10-20 per class is enough to see
a more realistic, imperfect evaluation report — which is actually what Step 6 wants to see:
a model that makes *some* mistakes, with a confusion matrix showing what it confuses).

To use your own data: put real files' extracted text into `data/dataset.csv` with the same
`text,label` columns (label ∈ Invoice/Resume/Other), then re-run `train_classifier.py`.

## What changed from Week 2 → Week 3

| Area | Week 2 | Week 3 |
|---|---|---|
| Text cleaning | None — raw PyMuPDF/OCR output used directly | `preprocess.py`: unicode normalization, control-char stripping, hyphenated line-break repair, whitespace collapsing, usability check |
| OCR | Basic OCR fallback | Image preprocessing before OCR (grayscale, upscaling, denoising, adaptive threshold) with a raw-image retry if preprocessing hurts a clean scan; scanned-PDF pages detected per-page (not whole-doc) |
| Classification | Rule-based keyword matching only | Rule-based kept as baseline; TF-IDF + Logistic Regression trained and compared against Linear SVM and Naive Bayes; best model auto-selected by macro F1 |
| Evaluation | None | Full accuracy/precision/recall/F1 + confusion matrix, computed on a held-out test split, saved to `models/evaluation_report.json` and rendered in-app |
| Field extraction | Simple regexes | Hardened patterns (invoice number requires a digit and isn't fooled by a bare "INVOICE" header; skills section is anchored to its own heading line so it isn't fooled by the word "skills" mid-sentence); date/total-amount extraction now looks for the value nearest the relevant label first |
| Missing fields | Not handled explicitly | Every field always resolves to a value or `"Not Found"` — never blank, never a crash; missing fields are surfaced in the UI |
| Confidence | None | Shown via `predict_proba` when the underlying model supports it; never fabricated when unavailable |
| Testing | 3 sample documents | `tests/generate_sample_docs.py` produces native PDFs, an OCR-only scanned image, a noisy/rotated scan, a blank image, and a deliberately corrupt PDF — all verified not to crash the pipeline |

## Testing evidence

Ran the full pipeline (`extract_text -> clean_text -> classify_document -> extract_fields`)
against all fixtures in `sample_docs/`:

| File | Extraction method | Classified as | Missing fields |
|---|---|---|---|
| native_invoice.pdf | Direct PDF text | Invoice ✅ | none |
| native_resume.pdf | Direct PDF text | Resume ✅ | none |
| scanned_invoice.png | OCR | Invoice ✅ | Invoice Number (OCR misread it — realistic OCR limitation) |
| noisy_scan.png (rotated + noise) | OCR | Invoice ✅ | none |
| blank.png | OCR attempted | — | flagged as "no usable text", no crash |
| corrupt.pdf | — | — | flagged as unreadable, no crash |

Also ran 120 synthetic samples (40 each of Invoice/Resume/Other) through classification +
extraction: 0 misclassifications, 0 missing fields on clean synthetic text (see caveat
above about synthetic data being easy — validate on real documents too).

## Week 3 Completion Checklist

- [x] Prepared and reviewed a cleaner dataset (synthetic generator; swap in real docs before submitting)
- [x] Cleaned and normalized extracted text
- [x] Tested OCR with scanned/image documents (incl. noisy/rotated)
- [x] Trained and evaluated a simple classifier
- [x] Compared suitable models (Logistic Regression, Linear SVM, Naive Bayes)
- [x] Improved information extraction
- [x] Handled missing fields
- [x] Added confidence information where appropriate
- [x] Tested the updated MVP (native PDFs, scans, blank, corrupt)
- [x] Updated GitHub and README (this file)
