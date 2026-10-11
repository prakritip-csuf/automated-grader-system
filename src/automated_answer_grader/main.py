"""FastAPI application entry point."""

from fastapi import FastAPI

from automated_answer_grader.api.router import api_router
from automated_answer_grader.core.config import (
    API_V1_PREFIX,
    APP_NAME,
    APP_VERSION,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    application = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description=(
            "API for an OCR and NLP-based automated short-answer "
            "grading system."
        ),
    )

    application.include_router(
        api_router,
        prefix=API_V1_PREFIX,
    )

    @application.get("/", tags=["meta"])
    async def root() -> dict[str, str]:
        """Return basic service metadata."""

        return {
            "name": APP_NAME,
            "version": APP_VERSION,
            "docs": "/docs",
        }

    return application


app = create_app()