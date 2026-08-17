"""Health-check API route."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from automated_grader_system.core.config import APP_SLUG, APP_VERSION


class HealthResponse(BaseModel):
    """Response returned when the API is operational."""

    status: Literal["ok"]
    service: str
    version: str


router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check whether the API is running",
)
async def health_check() -> HealthResponse:
    """Return the current API health state."""

    return HealthResponse(
        status="ok",
        service=APP_SLUG,
        version=APP_VERSION,
    )