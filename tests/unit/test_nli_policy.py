import pytest

from automated_answer_grader.domain.schemas import GradeRequest, RubricCriterion
from automated_answer_grader.pipelines.grading.semantic import (
    SemanticGrader,
    SimilarityMatch,
)


class FakeNLI:
    def __init__(self, contradiction: float) -> None:
        self.contradiction = contradiction

    def probabilities(self, premise: str, hypothesis: str) -> dict[str, float]:
        return {
            "contradiction": self.contradiction,
            "entailment": 1 - self.contradiction,
            "neutral": 0.0,
        }


def make_grader(nli, policy="review_only") -> SemanticGrader:
    # Bypass model initialization for deterministic unit tests.
    grader = object.__new__(SemanticGrader)
    grader.partial_threshold = 0.20
    grader.full_threshold = 0.80
    grader.contradiction_threshold = 0.60
    grader.nli = nli
    grader.nli_policy = policy
    grader._best_match = lambda units, refs: SimilarityMatch(
        score=0.90, student_unit=units[0], reference=refs[0]
    )
    return grader


def request() -> GradeRequest:
    return GradeRequest(
        question="Do plants need sunlight?",
        reference_answer="Plants require sunlight for photosynthesis.",
        student_answer="Plants do not require sunlight for photosynthesis.",
        criteria=[RubricCriterion(
            id="light", description="Light is required", points=2,
            references=["Plants require sunlight for photosynthesis."],
        )],
    )


def test_review_only_preserves_points_and_flags() -> None:
    result = make_grader(FakeNLI(0.95)).grade(request())
    assert result.score == 2.0
    assert result.review_required
    assert result.criteria[0].contradiction_probability == pytest.approx(0.95)
    assert any("contradiction" in reason for reason in result.review_reasons)


def test_penalty_reduces_only_matched_criterion() -> None:
    result = make_grader(FakeNLI(0.95), "penalty").grade(request())
    assert result.score == 0.0
    assert result.review_required


def test_no_contradiction_keeps_score() -> None:
    result = make_grader(FakeNLI(0.10), "penalty").grade(request())
    assert result.score == 2.0
    assert not result.review_required


def test_no_nli_preserves_baseline() -> None:
    result = make_grader(None).grade(request())
    assert result.score == 2.0
    assert result.criteria[0].contradiction_probability is None


def test_blank_answer_is_zero_and_reviewed() -> None:
    req = request().model_copy(update={"student_answer": "   "})
    result = make_grader(None).grade(req)
    assert result.score == 0
    assert result.review_required