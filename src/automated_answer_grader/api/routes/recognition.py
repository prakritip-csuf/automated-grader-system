from functools import lru_cache

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from automated_answer_grader.pipelines.preprocessing.handwriting import (
    load_rgb_image,
)

from automated_answer_grader.pipelines.recognition.trocr import (
    TrOCRRecognizer,
)

router = APIRouter(tags=["recognition"])


@lru_cache(maxsize=1)
def get_recognizer() -> TrOCRRecognizer:
    return TrOCRRecognizer()


# handwriting recognition endpoint
@router.post("/handwriting")
async def recognize_handwriting(
    file: UploadFile = File(...),
) -> dict[str, object]:

    allowed_types = {
        "image/png",
        "image/jpeg",
        "image/jpg",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Only PNG and JPEG images are supported.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    try:
        # convert image bytes into a PIL RGB image
        image = load_rgb_image(image_bytes)

        recognizer = get_recognizer()

        result = recognizer.recognize_answer(image)

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc

    return {
        "filename": file.filename or "unknown",
        "recognized_text": result.text,
        "lines": result.lines,
        "line_count": len(result.lines),
    }
