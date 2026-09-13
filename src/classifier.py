"""
classifier.py
-------------
Two classification strategies, as required by Week 3:

1. Rule-based baseline (kept from Week 2 logic, cleaned up) — fast, zero
   training required, used as a sanity-check comparison.
2. ML model (TF-IDF + Logistic Regression, or whichever model
   `models/train_classifier.py` decided was best) — loaded from disk at
   inference time, with a confidence score.

The app calls `classify_document()`, which prefers the trained ML model but
falls back to the rule-based baseline if no trained model is found on disk
(so the app never breaks if someone hasn't run training yet).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

LABELS = ["Invoice", "Resume", "Other"]


@dataclass
class ClassificationResult:
    label: str
    confidence: float | None   # None when not meaningfully available
    method: str                 # "rule_based" or f"ml:{model_name}"


# ---------------------------------------------------------------------------
# Rule-based baseline (Week 2 approach, kept as a comparison point)
# ---------------------------------------------------------------------------

INVOICE_KEYWORDS = ["invoice", "bill to", "total amount", "subtotal", "invoice number",
                     "amount due", "tax", "purchase order"]
RESUME_KEYWORDS = ["resume", "curriculum vitae", "objective", "skills", "experience",
                    "education", "references", "career summary"]


def rule_based_classify(cleaned_text_lower: str) -> ClassificationResult:
    invoice_score = sum(1 for kw in INVOICE_KEYWORDS if kw in cleaned_text_lower)
    resume_score = sum(1 for kw in RESUME_KEYWORDS if kw in cleaned_text_lower)

    if invoice_score == 0 and resume_score == 0:
        return ClassificationResult(label="Other", confidence=None, method="rule_based")
    if invoice_score >= resume_score:
        return ClassificationResult(label="Invoice", confidence=None, method="rule_based")
    return ClassificationResult(label="Resume", confidence=None, method="rule_based")


# ---------------------------------------------------------------------------
# ML model (loaded from disk, trained by models/train_classifier.py)
# ---------------------------------------------------------------------------

_vectorizer = None
_model = None
_model_name = None


def _load_ml_assets() -> bool:
    """Lazily loads the trained vectorizer + model. Returns False (and never
    raises) if they haven't been trained yet."""
    global _vectorizer, _model, _model_name
    if _model is not None:
        return True
    if not (VECTORIZER_PATH.exists() and MODEL_PATH.exists()):
        return False
    try:
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _model = joblib.load(MODEL_PATH)
        if METADATA_PATH.exists():
            _model_name = json.loads(METADATA_PATH.read_text()).get("best_model", "unknown")
        else:
            _model_name = "unknown"
        return True
    except Exception:  # noqa: BLE001
        _vectorizer, _model, _model_name = None, None, None
        return False


def ml_classify(normalized_text: str) -> ClassificationResult | None:
    """Returns None if no trained model is available (caller should fall
    back to the rule-based baseline in that case)."""
    if not _load_ml_assets():
        return None

    features = _vectorizer.transform([normalized_text])
    label = _model.predict(features)[0]

    confidence = None
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(features)[0]
        confidence = float(max(proba))

    return ClassificationResult(label=label, confidence=confidence, method=f"ml:{_model_name}")


def classify_document(cleaned_text: str, normalized_text: str) -> ClassificationResult:
    """Main entry point. Tries the trained ML model first, falls back to
    the rule-based baseline so the app always returns *something*."""
    if not normalized_text.strip():
        return ClassificationResult(label="Other", confidence=None, method="none (empty text)")

    ml_result = ml_classify(normalized_text)
    if ml_result is not None:
        return ml_result

    return rule_based_classify(cleaned_text.lower())
