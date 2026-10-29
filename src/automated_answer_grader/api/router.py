"""Top-level API router."""

from fastapi import APIRouter

from automated_answer_grader.api.routes.extraction import router as extraction_router
from automated_answer_grader.api.routes.grading import router as grading_router
from automated_answer_grader.api.routes.health import router as health_router
from automated_answer_grader.api.routes.recognition import router as recognition_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(extraction_router)
api_router.include_router(recognition_router, prefix="/recognition")
api_router.include_router(grading_router)
