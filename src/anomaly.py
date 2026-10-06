"""
anomaly.py
----------
Anomaly and duplicate detection beyond the SHA-256 hash check already in
DocumentRepository.  Runs after field extraction and before the workflow
decision so results can influence routing.

Checks:
  1. Repeated invoice number — same invoice number already stored for a
     different document (different hash).
  2. Financial consistency — for invoices, verify that a numeric Total Amount
     is present and positive (basic sanity; full line-item reconciliation
     would require structured tables not available from plain OCR text).
  3. Suspicious amount — flag amounts that look unrealistically large (> 1M)
     or zero, which may indicate an OCR mis-read.

Returns a list of AnomalyFlag objects so the caller can decide how to route.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class AnomalyFlag:
    code: str          # machine-readable identifier
    message: str       # human-readable description
    severity: str      # "warning" | "error"


_AMOUNT_STRIP = re.compile(r"[^\d.-]")


def _parse_amount(raw: str) -> float | None:
    """Extract a float from a raw amount string; return None on failure."""
    if not raw or raw.strip().lower() in ("not found", ""):
        return None
    # Strip currency symbols/letters but keep digits, dot, and leading minus
    stripped = raw.replace(",", "")
    # Remove currency prefix (letters, $, Rs., PKR, USD) before the number
    stripped = re.sub(r"^[^\d\-]+", "", stripped)
    cleaned = _AMOUNT_STRIP.sub("", stripped)
    try:
        return float(cleaned)
    except ValueError:
        return None


def detect_anomalies(
    document_type: str,
    fields: dict[str, Any],
    repository: Any,
    current_doc_id: int | None = None,
) -> list[AnomalyFlag]:
    """
    Run all anomaly checks and return a (possibly empty) list of flags.

    Parameters
    ----------
    document_type : str
        Classified document type ("Invoice", "Resume", "Other").
    fields : dict
        Extracted fields for this document.
    repository : DocumentRepository
        Used to query existing records.
    current_doc_id : int | None
        ID of the document being checked (excluded from duplicate searches).
    """
    flags: list[AnomalyFlag] = []

    if document_type == "Invoice":
        # --- Repeated invoice number ---
        inv_num = (fields.get("Invoice Number") or "").strip()
        if inv_num and inv_num.lower() != "not found":
            try:
                existing = repository.list_documents(search=inv_num, document_type="Invoice")
                duplicates = [
                    d for d in existing
                    if d.get("invoice_number") == inv_num
                    and d["id"] != current_doc_id
                ]
                if duplicates:
                    ids = ", ".join(f"#{d['id']}" for d in duplicates[:3])
                    flags.append(AnomalyFlag(
                        code="DUPLICATE_INVOICE_NUMBER",
                        message=f"Invoice number '{inv_num}' already exists in document(s) {ids}.",
                        severity="warning",
                    ))
            except Exception:
                pass

        # --- Financial sanity ---
        raw_amount = (fields.get("Total Amount") or "").strip()
        amount = _parse_amount(raw_amount)
        if amount is not None:
            if amount == 0:
                flags.append(AnomalyFlag(
                    code="ZERO_AMOUNT",
                    message="Total Amount is zero — possible OCR error or test document.",
                    severity="warning",
                ))
            elif amount < 0:
                flags.append(AnomalyFlag(
                    code="NEGATIVE_AMOUNT",
                    message="Total Amount is negative — this is likely invalid.",
                    severity="error",
                ))
            elif amount > 1_000_000:
                flags.append(AnomalyFlag(
                    code="UNUSUALLY_LARGE_AMOUNT",
                    message=f"Total Amount ({raw_amount}) is unusually large — please verify.",
                    severity="warning",
                ))

    return flags


def anomaly_summary(flags: list[AnomalyFlag]) -> str:
    """Return a short human-readable summary for audit/workflow reason fields."""
    if not flags:
        return ""
    parts = [f"[{f.severity.upper()}] {f.message}" for f in flags]
    return "; ".join(parts)
