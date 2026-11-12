from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from automated_answer_grader.core.dependencies import (
    get_trocr_recognizer,
)
from automated_answer_grader.pipelines.recognition.pdf import (
    PDFProcessingError,
    extract_answer_from_pdf,
)

router = APIRouter(
    prefix="/extraction",
    tags=["extraction"],
)

MAX_PDF_BYTES = 10 * 1024 * 1024


@router.post("/pdf")
async def extract_pdf(
    file: UploadFile = File(...),
) -> dict[str, str]:
    """Extract text from a digital or scanned answer PDF."""

    # Validate the uploaded file type.
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported.",
        )

    # Read the uploaded PDF with a size limit.
    pdf_bytes = await file.read(MAX_PDF_BYTES + 1)

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty.",
        )

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The PDF must be 10 MB or smaller.",
        )

    # Extract embedded text or use TrOCR fallback.
    try:
        text, method = await run_in_threadpool(
            extract_answer_from_pdf,
            pdf_bytes,
            get_trocr_recognizer,
        )

    except PDFProcessingError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # Check whether any text was extracted.
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No answer text could be extracted.",
        )

    # Return the extraction result.
    return {
        "text": text,
        "method": method,
    }

