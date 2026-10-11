"""Top-level API router."""

from fastapi import APIRouter

from automated_answer_grader.api.routes.health import router as health_router

from automated_answer_grader.api.routes.recognition import (
    router as recognition_router,
)

api_router = APIRouter()
api_router.include_router(health_router)

api_router.include_router(recognition_router, prefix="/recognition")
