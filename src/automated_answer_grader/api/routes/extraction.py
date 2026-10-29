from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from automated_answer_grader.core.dependencies import get_trocr_recognizer
from automated_answer_grader.pipelines.recognition.pdf import (
    extract_answer_from_pdf,
)

from automated_answer_grader.pipelines.recognition.trocr import TrOCRRecognizer

router = APIRouter(prefix="/extraction", tags=["extraction"])

@router.post("/pdf")
async def extract_pdf(
    file: UploadFile = File(...),
    recognizer: TrOCRRecognizer = Depends(get_trocr_recognizer)
) -> dict[str, str]:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Upload a PDF File.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="The PDF is empty.")

    text, method = extract_answer_from_pdf(pdf_bytes, recognizer)
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No answer text could be extracted.",
        )

    return {"text": text, "method": method}
    