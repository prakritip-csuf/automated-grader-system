"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from automated_answer_grader.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide a FastAPI test client."""

    with TestClient(app) as test_client:
        yield test_client
