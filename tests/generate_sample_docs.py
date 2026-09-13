"""
generate_sample_docs.py
------------------------
Creates a handful of realistic test fixtures so the pipeline can be tested
end-to-end without needing to source real documents first:

- native_invoice.pdf   -> normal PDF with a real text layer
- native_resume.pdf    -> normal PDF with a real text layer
- scanned_invoice.png  -> image-only "scan" of an invoice (forces OCR path)
- noisy_scan.png       -> scanned image with added noise/rotation (harder OCR)
- blank.png            -> blank image (tests the "no usable text" path)
- corrupt.pdf          -> a deliberately broken file (tests error handling)

Run: python tests/generate_sample_docs.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.generate_dataset import make_invoice, make_resume  # noqa: E402

OUT_DIR = ROOT / "sample_docs"
OUT_DIR.mkdir(exist_ok=True)

random.seed(11)


def make_native_pdf(text: str, out_path: Path) -> None:
    c = canvas.Canvas(str(out_path), pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica", 11)
    for line in text.split("\n"):
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - 50
        c.drawString(50, y, line[:110])
        y -= 16
    c.save()


def text_to_image(text: str, size=(1000, 1300)) -> Image.Image:
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Image if False else ImageDraw.Draw(img)
    font = None
    for font_path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/verdana.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, 22)
            break
        except Exception:  # noqa: BLE001
            continue
    if font is None:
        font = ImageFont.load_default()

    y = 40
    for line in text.split("\n"):
        draw.text((40, y), line, fill="black", font=font)
        y += 32
    return img


def add_scan_noise(img: Image.Image, rotate_deg: float = 1.5) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 12, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy = Image.fromarray(arr)
    return noisy.rotate(rotate_deg, expand=True, fillcolor="white")


def main() -> None:
    invoice_text = make_invoice()
    resume_text = make_resume()

    make_native_pdf(invoice_text, OUT_DIR / "native_invoice.pdf")
    make_native_pdf(resume_text, OUT_DIR / "native_resume.pdf")
    print("Wrote native_invoice.pdf, native_resume.pdf (real text layer)")

    scanned = text_to_image(invoice_text)
    scanned.save(OUT_DIR / "scanned_invoice.png")
    print("Wrote scanned_invoice.png (image-only, forces OCR)")

    noisy = add_scan_noise(text_to_image(invoice_text), rotate_deg=1.2)
    noisy.save(OUT_DIR / "noisy_scan.png")
    print("Wrote noisy_scan.png (noise + slight rotation, harder OCR case)")

    blank = Image.new("RGB", (800, 1000), color="white")
    blank.save(OUT_DIR / "blank.png")
    print("Wrote blank.png (no text at all)")

    (OUT_DIR / "corrupt.pdf").write_bytes(b"%PDF-1.4 not a real pdf body ---")
    print("Wrote corrupt.pdf (deliberately broken file)")

    print(f"\nAll sample docs in {OUT_DIR}/ — use these to test the Streamlit app.")


if __name__ == "__main__":
    main()
