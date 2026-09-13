"""
field_extraction.py
--------------------
Regex/keyword-based field extraction, improved from Week 2:
- Invoice: Invoice Number, Date, Company Name, Total Amount.
- Resume:  Name, Email, Phone, Skills.

Design choice per the Week 3 brief: keep this simple and understandable
(no NLP/NER model) but make the patterns robust to real-world formatting
variance. Every field falls back to "Not Found" instead of raising or
leaving a blank, and every miss is logged so it's easy to see what to
improve next (see get_missing_fields()).
"""

from __future__ import annotations

import re

NOT_FOUND = "Not Found"

# --- shared patterns -------------------------------------------------------

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"
)
DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
MONEY_RE = re.compile(
    r"(?:Rs\.?|PKR|USD|\$)\s?[\d,]+(?:\.\d{1,2})?|\b[\d,]{4,}(?:\.\d{1,2})?\b"
)


# --- invoice fields ----------------------------------------------------------

def extract_invoice_number(text: str) -> str:
    # Require "no./number/#" (or a colon right after "invoice") so we don't
    # accidentally capture the word "Invoice" itself off a bare document
    # title line like "INVOICE" with no number nearby.
    patterns = [
        r"invoice\s*(?:no\.?|number|num|#)\s*[:\-]?\s*([A-Za-z0-9/\-]{3,20})",
        r"invoice\s*[:\-]\s*([A-Za-z0-9/\-]{3,20})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            candidate = match.group(1).strip()
            if any(ch.isdigit() for ch in candidate) and candidate.lower() != "invoice":
                return candidate
    return NOT_FOUND


def extract_date(text: str) -> str:
    # Prefer a date that appears near the word "date" to avoid grabbing an
    # unrelated number sequence.
    for line in text.splitlines():
        if re.search(r"\bdate\b", line, re.IGNORECASE):
            m = DATE_RE.search(line)
            if m:
                return m.group(0)
    m = DATE_RE.search(text)
    return m.group(0) if m else NOT_FOUND


def extract_company_name(text: str) -> str:
    for pattern in [
        r"(?:bill\s*to|company\s*name|company|vendor|from)\s*[:\-]\s*(.+)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().split("\n")[0]
            candidate = re.sub(r"\s{2,}.*$", "", candidate)  # drop trailing junk
            if candidate:
                return candidate
    return NOT_FOUND


def extract_total_amount(text: str) -> str:
    for line in text.splitlines():
        if re.search(r"\btotal\b", line, re.IGNORECASE) and not re.search(r"subtotal", line, re.IGNORECASE):
            m = MONEY_RE.search(line)
            if m:
                return m.group(0).strip()
    # fallback: last money-like value in the document (often the grand total)
    matches = MONEY_RE.findall(text)
    return matches[-1].strip() if matches else NOT_FOUND


def extract_invoice_fields(text: str) -> dict:
    return {
        "Invoice Number": extract_invoice_number(text),
        "Date": extract_date(text),
        "Company Name": extract_company_name(text),
        "Total Amount": extract_total_amount(text),
    }


# --- resume fields ------------------------------------------------------------

def extract_name(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[:6]:
        low = line.lower()
        if any(kw in low for kw in ["resume", "curriculum vitae", "cv", "email", "phone", "@"]):
            continue
        words = line.split()
        if 1 < len(words) <= 4 and all(w[0].isupper() for w in words if w[0].isalpha()):
            return line
    return NOT_FOUND


def extract_email(text: str) -> str:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else NOT_FOUND


def extract_phone(text: str) -> str:
    for line in text.splitlines():
        if re.search(r"\bphone\b|\bcontact\b|\bmobile\b|\btel\b", line, re.IGNORECASE):
            m = PHONE_RE.search(line)
            if m and len(re.sub(r"\D", "", m.group(0))) >= 7:
                return m.group(0).strip()
    m = PHONE_RE.search(text)
    if m and len(re.sub(r"\D", "", m.group(0))) >= 7:
        return m.group(0).strip()
    return NOT_FOUND


SKILL_VOCAB = [
    "python", "java", "javascript", "sql", "c++", "c#", "react", "node.js", "aws",
    "docker", "kubernetes", "machine learning", "deep learning", "tensorflow",
    "pytorch", "excel", "power bi", "data analysis", "project management",
    "communication", "leadership", "sales", "marketing", "accounting", "linux",
    "git", "html", "css", "flask", "django", "scikit-learn", "pandas", "numpy",
]


def extract_skills(text: str) -> str:
    low = text.lower()

    # Preferred: an explicit "Skills" section heading on its own line —
    # anchored to line-start so we don't match the word "skills" appearing
    # mid-sentence elsewhere (e.g. "...apply my analytical skills.").
    section_match = re.search(
        r"^\s*skills\s*:?\s*$\n+(.+?)(?:\n\s*\n|\n\s*(?:experience|education|references|projects)\b|\Z)",
        low, re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if section_match:
        section = section_match.group(1)
        found = sorted({skill for skill in SKILL_VOCAB if skill in section})
        if found:
            return ", ".join(s.title() if s.islower() else s for s in found)
        # No known vocab matched, but there IS a skills section — return
        # a cleaned comma/bullet-split snippet instead of giving up.
        parts = re.split(r"[,\u2022\n]", section)
        parts = [p.strip(" -\t") for p in parts if p.strip(" -\t")]
        if parts:
            return ", ".join(parts[:8])

    # Fallback: scan the whole doc for known skill keywords.
    found = sorted({skill for skill in SKILL_VOCAB if skill in low})
    if found:
        return ", ".join(s.title() if s.islower() else s for s in found)
    return NOT_FOUND


def extract_resume_fields(text: str) -> dict:
    return {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Phone": extract_phone(text),
        "Skills": extract_skills(text),
    }


# --- dispatch ------------------------------------------------------------------

def extract_fields(text: str, doc_type: str) -> dict:
    if doc_type == "Invoice":
        return extract_invoice_fields(text)
    if doc_type == "Resume":
        return extract_resume_fields(text)
    return {}


def get_missing_fields(fields: dict) -> list[str]:
    return [name for name, value in fields.items() if value == NOT_FOUND]
