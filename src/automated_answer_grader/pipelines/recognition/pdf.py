from __future__ import annotations

import io
from collections.abc import Callable
from typing import TYPE_CHECKING

import pymupdf
from PIL import Image

if TYPE_CHECKING:
    from automated_answer_grader.pipelines.recognition.trocr import (
        TrOCRRecognizer,
    )

MAX_PDF_PAGES = 5
RENDER_SCALE = 2.0


class PDFProcessingError(ValueError):
    """Raised when a PDF cannot be processed safely."""


def _open_pdf(pdf_bytes: bytes) -> pymupdf.Document:
    """Open a PDF and validate its basic properties."""

    if not pdf_bytes:
        raise PDFProcessingError("The PDF file is empty.")

    try:
        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        )
    except (pymupdf.FileDataError, ValueError, RuntimeError) as exc:
        raise PDFProcessingError(
            "The uploaded file is not a valid PDF."
        ) from exc

    if document.needs_pass:
        document.close()
        raise PDFProcessingError(
            "Password-protected PDFs are not supported."
        )

    if document.page_count == 0:
        document.close()
        raise PDFProcessingError(
            "The PDF does not contain any pages."
        )

    if document.page_count > MAX_PDF_PAGES:
        document.close()
        raise PDFProcessingError(
            f"PDFs must contain at most {MAX_PDF_PAGES} pages."
        )

    return document


def extract_selectable_text(pdf_bytes: bytes) -> str:
    """Extract embedded text from a digital PDF."""

    with _open_pdf(pdf_bytes) as document:
        pages = [
            page.get_text(sort=True).strip()
            for page in document
        ]

    return "\n\n".join(
        text for text in pages if text
    ).strip()



def _render_page(
    page: pymupdf.Page,
    scale: float = RENDER_SCALE,
) -> Image.Image:
    """Convert a PDF page into an RGB image."""

    if not 0 < scale <= 3:
        raise ValueError(
            "Render scale must be greater than 0 and at most 3."
        )

    matrix = pymupdf.Matrix(scale, scale)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image_bytes = pixmap.tobytes("png")

    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def render_pdf_pages(
    pdf_bytes: bytes,
    *,
    scale: float = RENDER_SCALE,
) -> list[Image.Image]:
    """Render every page as a separate RGB image."""

    with _open_pdf(pdf_bytes) as document:
        images = [
            _render_page(page, scale)
            for page in document
        ]

    return images

def extract_answer_from_pdf(
    pdf_bytes: bytes,
    recognizer_factory: Callable[[], TrOCRRecognizer],
) -> tuple[str, str]:
    """Extract text, using TrOCR only for image-only pages.

    Returns:
        (extracted_text, extraction_method)

    Methods:
        embedded_text: All pages used direct text extraction.
        trocr_ocr: All pages required TrOCR.
        mixed: Both extraction methods were used.
    """

    extracted_pages: list[str] = []
    methods: list[str] = []

    # Model is created only if a scanned page needs OCR.
    recognizer = None

    with _open_pdf(pdf_bytes) as document:

        for page in document:

            direct_text = page.get_text(
                sort=True
            ).strip()

            if direct_text:
                extracted_pages.append(direct_text)
                methods.append("embedded_text")
                continue

            image = _render_page(page)

            if recognizer is None:
                recognizer = recognizer_factory()


            result = recognizer.recognize_answer(image)

            recognized_text = result.text.strip()

            if recognized_text:
                extracted_pages.append(recognized_text)

            methods.append("trocr_ocr")

    complete_text = "\n\n".join(extracted_pages).strip()

    if not methods:
        method = "unknown"
    elif len(set(methods)) == 1:
        method = methods[0]
    else:
        method = "mixed"

    return complete_text, method
