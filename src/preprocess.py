"""
preprocess.py
-------------
Cleans raw extracted text (from PyMuPDF or OCR) before it's fed into the
classifier or the field-extraction regexes.

OCR/PDF extraction commonly produces:
- repeated blank lines / excessive whitespace
- broken words split across lines with hyphens
- stray non-printable / control characters
- inconsistent casing and punctuation spacing

None of this should ever throw — worst case, it returns an empty string.
"""

from __future__ import annotations

import re
import unicodedata

MIN_USABLE_LENGTH = 15  # fewer real characters than this = "empty/unusable"


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def remove_control_characters(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")


def fix_hyphenated_linebreaks(text: str) -> str:
    """'invoice-\nnumber' -> 'invoicenumber' (common in wrapped OCR/PDF text)."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse 3+ blank lines to 1
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def clean_text(raw_text: str) -> str:
    """Full cleaning pipeline. Safe on None/empty input."""
    if not raw_text:
        return ""
    text = normalize_unicode(raw_text)
    text = remove_control_characters(text)
    text = fix_hyphenated_linebreaks(text)
    text = collapse_whitespace(text)
    return text


def is_usable(cleaned_text: str) -> bool:
    """True if there's enough real content to bother classifying/extracting."""
    letters_and_digits = sum(ch.isalnum() for ch in cleaned_text)
    return letters_and_digits >= MIN_USABLE_LENGTH


def normalize_for_classification(cleaned_text: str) -> str:
    """Extra normalization specific to the classifier (lowercase, strip
    excess punctuation) — kept separate from clean_text() because the
    field-extraction step still wants original casing (e.g. for names)."""
    text = cleaned_text.lower()
    text = re.sub(r"[^a-z0-9\s@.,:/#-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
