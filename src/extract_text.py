"""
extract_text.py
----------------
Handles reading raw text out of uploaded documents.

Strategy:
1. If it's a PDF, try PyMuPDF (fitz) direct text extraction first — it's fast
   and accurate for "born-digital" PDFs.
2. If that yields little/no text (i.e. it's a scanned PDF), OR the file is an
   image (JPG/PNG), fall back to OCR (Tesseract via pytesseract).
3. Before OCR, run light image preprocessing (grayscale, resize, threshold,
   denoise) to improve OCR accuracy on messy scans.

Every function here is defensive: it should never raise on a bad/corrupt
file — it should return an empty string + an error note instead, so the
Streamlit app never crashes on upload.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import os

import cv2
import numpy as np
import pytesseract
from PIL import Image

# Point pytesseract at the Tesseract binary on Windows if it's not on PATH.
_TESSERACT_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.isfile(_TESSERACT_WIN):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_WIN

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

logger = logging.getLogger(__name__)

# Below this many "real" characters per page, we treat a PDF page as scanned
# (i.e. no usable embedded text layer) and route it through OCR instead.
MIN_CHARS_PER_PAGE_THRESHOLD = 20


@dataclass
class ExtractionResult:
    """Everything downstream code needs to know about how text was pulled out."""

    text: str = ""
    method: str = "none"          # "pymupdf", "ocr", "ocr_fallback", "none"
    used_ocr: bool = False
    page_count: int = 0
    warnings: list[str] = field(default_factory=list)
    success: bool = False


def _preprocess_image_for_ocr(pil_image: Image.Image) -> np.ndarray:
    """Grayscale -> denoise -> adaptive threshold. Cheap, beginner-friendly,
    and it measurably helps OCR on low-quality scans/phone photos."""
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscale small images — Tesseract does much better above ~300 DPI equiv.
    h, w = img.shape
    if max(h, w) < 1500:
        scale = 1500 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    img = cv2.fastNlMeansDenoising(img, h=10)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return img


def _ocr_image(pil_image: Image.Image) -> str:
    try:
        processed = _preprocess_image_for_ocr(pil_image)
        text = pytesseract.image_to_string(processed)
        if len(text.strip()) < 5:
            # Preprocessing sometimes hurts clean images — retry on raw image.
            text = pytesseract.image_to_string(pil_image)
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR failed: %s", exc)
        return ""


def extract_from_image_bytes(file_bytes: bytes) -> ExtractionResult:
    result = ExtractionResult(page_count=1)
    try:
        pil_image = Image.open(io.BytesIO(file_bytes))
        text = _ocr_image(pil_image)
        result.text = text
        result.method = "ocr"
        result.used_ocr = True
        result.success = True
        if len(text.strip()) < 5:
            result.warnings.append("OCR returned almost no text — image may be blank, "
                                    "too low-resolution, or not a document.")
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"Could not read image file: {exc}")
    return result


def extract_from_pdf_bytes(file_bytes: bytes) -> ExtractionResult:
    result = ExtractionResult()

    if fitz is None:
        result.warnings.append("PyMuPDF not installed — cannot process PDFs.")
        return result

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"Could not open PDF (corrupted or unsupported): {exc}")
        return result

    result.page_count = doc.page_count
    native_text_parts: list[str] = []
    needs_ocr_pages: list[int] = []

    for page_index, page in enumerate(doc):
        try:
            page_text = page.get_text("text")
        except Exception as exc:  # noqa: BLE001
            page_text = ""
            result.warnings.append(f"Page {page_index + 1}: text extraction error ({exc})")

        native_text_parts.append(page_text)
        if len(page_text.strip()) < MIN_CHARS_PER_PAGE_THRESHOLD:
            needs_ocr_pages.append(page_index)

    native_text = "\n".join(native_text_parts)

    # Whole document has almost no embedded text -> it's a scanned PDF.
    all_pages_need_ocr = len(needs_ocr_pages) == doc.page_count

    if not all_pages_need_ocr and native_text.strip():
        result.text = native_text
        result.method = "pymupdf"
        result.used_ocr = False
        result.success = True
        doc.close()
        return result

    # Fall back to OCR, rendering each page (or just the empty ones) to an image.
    ocr_text_parts: list[str] = []
    for page_index, page in enumerate(doc):
        if native_text_parts[page_index].strip() and page_index not in needs_ocr_pages:
            ocr_text_parts.append(native_text_parts[page_index])
            continue
        try:
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_bytes))
            ocr_text_parts.append(_ocr_image(pil_image))
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"Page {page_index + 1}: OCR render failed ({exc})")

    doc.close()

    result.text = "\n".join(ocr_text_parts)
    result.method = "ocr_fallback"
    result.used_ocr = True
    result.success = True
    if len(result.text.strip()) < 5:
        result.warnings.append("Document appears to be a scanned PDF but OCR "
                                "could not recover readable text.")
    return result


def extract_text(file_bytes: bytes, file_extension: str) -> ExtractionResult:
    """Single entry point used by the app. file_extension e.g. 'pdf', 'jpg', 'png'."""
    ext = file_extension.lower().lstrip(".")
    if ext == "pdf":
        return extract_from_pdf_bytes(file_bytes)
    if ext in {"jpg", "jpeg", "png"}:
        return extract_from_image_bytes(file_bytes)

    result = ExtractionResult()
    result.warnings.append(f"Unsupported file type: .{ext}")
    return result
