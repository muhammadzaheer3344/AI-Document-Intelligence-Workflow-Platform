"""Field validation rules for supported document types."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from src.field_extraction import NOT_FOUND

REQUIRED_FIELDS = {
    "Invoice": ("Invoice Number", "Date", "Company Name", "Total Amount"),
    "Resume": ("Name", "Email", "Skills"),
}
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_PATTERN = re.compile(r"^\+?[\d\s().-]{7,20}$")
AMOUNT_PATTERN = re.compile(r"^(?:Rs\.?|PKR|USD|\$)?\s*[\d,]+(?:\.\d{1,2})?$", re.IGNORECASE)
DATE_FORMATS = (
    "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y",
    "%m-%d-%Y", "%m-%d-%y", "%d-%m-%Y", "%d-%m-%y",
    "%Y-%m-%d", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y",
    "%b %d %Y", "%B %d %Y",
)


def validate_fields(document_type: str, fields: dict[str, Any]) -> dict[str, str]:
    """Return field-name to failure-reason mappings; an empty result is valid."""
    failures: dict[str, str] = {}
    for field_name in REQUIRED_FIELDS.get(document_type, ()):
        value = fields.get(field_name)
        if value is None or not str(value).strip() or str(value).strip().lower() == NOT_FOUND.lower():
            failures[field_name] = "Required field is missing."

    email = fields.get("Email")
    if email and str(email).strip().lower() != NOT_FOUND.lower() and not EMAIL_PATTERN.fullmatch(str(email).strip()):
        failures["Email"] = "Email format is invalid."

    phone = fields.get("Phone")
    if phone and str(phone).strip().lower() != NOT_FOUND.lower():
        digits = re.sub(r"\D", "", str(phone))
        if not PHONE_PATTERN.fullmatch(str(phone).strip()) or not 7 <= len(digits) <= 15:
            failures["Phone"] = "Phone format is invalid."

    date_value = fields.get("Date")
    if date_value and str(date_value).strip().lower() != NOT_FOUND.lower():
        parsed = False
        for date_format in DATE_FORMATS:
            try:
                datetime.strptime(str(date_value).strip(), date_format)
                parsed = True
                break
            except ValueError:
                continue
        if not parsed:
            failures["Date"] = "Date format is invalid or unrecognized."

    amount = fields.get("Total Amount")
    if amount and str(amount).strip().lower() != NOT_FOUND.lower():
        amount_text = str(amount).strip()
        numeric = re.sub(r"^(?:Rs\.?|PKR|USD|\$)\s*", "", amount_text, flags=re.IGNORECASE)
        if not AMOUNT_PATTERN.fullmatch(amount_text) or not re.search(r"\d", numeric):
            failures["Total Amount"] = "Amount must be a numeric value."
        else:
            try:
                if float(numeric.replace(",", "")) < 0:
                    failures["Total Amount"] = "Amount cannot be negative."
            except ValueError:
                failures["Total Amount"] = "Amount must be a numeric value."

    return failures