import io
from functools import lru_cache

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from automated_answer_grader.pipelines.recognition.trocr import (
    TrOCRRecognizer,
)


router = APIRouter(tags=["recognition"])


@lru_cache
def get_recognizer() -> TrOCRRecognizer:
    """Load the TrOCR model once and reuse it."""

    return TrOCRRecognizer()


@router.post("/handwriting")
async def recognize_handwriting(
    file: UploadFile = File(...),
) -> dict[str, str]:
    """Recognize handwritten text from an uploaded image."""

    allowed_types = {
        "image/png",
        "image/jpeg",
        "image/jpg",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPEG images are supported.",
        )

    try:
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        recognizer = get_recognizer()

        recognized_text = recognizer.recognize(image)

        return {
            "filename": file.filename or "unknown",
            "recognized_text": recognized_text,
        }

    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc
