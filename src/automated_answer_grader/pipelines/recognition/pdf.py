from __future__ import annotations

import io
import pymupdf
from PIL import Image

from automated_answer_grader.pipelines.recognition.trocr import (
    TrOCRRecognizer,
)

def extract_selectable_text(pdf_bytes:bytes) -> str:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        pages = [page.get_text(sort=True) for page in document]
    return "\n\n".join(pages).strip()

def render_pdf_pages(
    pdf_bytes: bytes,
    *,
    scale: float = 2.0,
) -> list[Image.Image]:
    images: list[Image.Image] = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        matrix = pymupdf.Matrix(scale, scale)
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            png_bytes = pixmap.tobytes("png")
            images.append(
                Image.open(io.BytesIO(png_bytes)).convert("RGB")
            )

    return images

def extract_answer_from_pdf(
    pdf_bytes: bytes,
    recognizer: TrOCRRecognizer,
) -> tuple[str, str]:
    direct_text = extract_selectable_text(pdf_bytes)

    if len(direct_text) >= 10:
        return direct_text, "embedded_text"

    page_images = render_pdf_pages(pdf_bytes)
    page_text = [
        recognizer.recognize_answer(image).text
        for image in page_images
    ]
    return "\n\n".join(text for text in page_text if text), "trocr_ocr"