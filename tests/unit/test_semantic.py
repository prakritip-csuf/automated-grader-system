import pytest

from automated_answer_grader.domain.schemas import (
    GradeRequest,
    RubricCriterion,
)

from automated_answer_grader.pipelines.grading.semantic import (
    SemanticGrader,
)


def make_grader_without_model() -> SemanticGrader:
    grader = object.__new__(SemanticGrader)

    grader.partial_threshold = 0.50
    grader.full_threshold = 0.72
    grader.contradiction_threshold = 0.60
    grader.nli = None

    return grader


def test_zero_credit() -> None:
    grader = make_grader_without_model()

    assert grader._points_fraction(0.30) == 0.0


def test_full_credit() -> None:
    grader = make_grader_without_model()

    assert grader._points_fraction(0.85) == 1.0


def test_partial_credit() -> None:
    grader = make_grader_without_model()

    assert grader._points_fraction(0.61) == pytest.approx(0.5)


def test_empty_answer() -> None:
    grader = make_grader_without_model()

    request = GradeRequest(
        question="What is photosynthesis?",
        reference_answer="Plants convert light energy.",
        student_answer="",
        criteria=[
            RubricCriterion(
                id="c1",
                description="Explains light energy",
                points=2,
                references=["Plants use light energy."],
            )
        ],
    )

    result = grader.grade(request)

    assert result.score == 0.0
    assert result.max_score == 2.0
    assert result.review_required is True
    assert result.review_reasons == ["Student answer is empty."]
