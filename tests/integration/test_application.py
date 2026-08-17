"""Integration tests for the initial FastAPI application."""

from fastapi.testclient import TestClient


def test_root_endpoint_returns_service_metadata(
    client: TestClient,
) -> None:
    """The root endpoint should identify the service."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": "Automated Answer Grader API",
        "version": "0.1.0",
        "docs": "/docs",
    }


def test_health_endpoint_reports_ok(
    client: TestClient,
) -> None:
    """The health endpoint should report an operational service."""

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "automated-grader-system",
        "version": "0.1.0",
    }