from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from automated_answer_grader.domain.schemas import (
    CriterionResult,
    GradeRequest,
    GradeResponse,
    RubricCriterion,
)
from automated_answer_grader.pipelines.grading.text import (
    normalize_text,
    split_answer_units,
)


@dataclass(frozen=True)
class SimilarityMatch:
    score: float
    student_unit: str
    reference: str


class SemanticGrader:
    """Score student answers against weighted rubric references."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        partial_threshold: float = 0.20,
        full_threshold: float = 0.80,
        contradiction_threshold: float = 0.60,
        use_nli: bool = False,
        nli_policy: Literal["review_only", "penalty"] = "review_only",
    ) -> None:
        if not 0 <= partial_threshold < full_threshold <= 1:
            raise ValueError("Require 0 <= partial_threshold < full_threshold <= 1.")
        if not 0 <= contradiction_threshold <= 1:
            raise ValueError("Contradiction threshold must be between 0 and 1.")
        if nli_policy not in ("review_only", "penalty"):
            raise ValueError("nli_policy must be 'review_only' or 'penalty'.")

        self.encoder = SentenceTransformer(embedding_model)
        self.partial_threshold = partial_threshold
        self.full_threshold = full_threshold
        self.contradiction_threshold = contradiction_threshold
        self.nli_policy = nli_policy
        self.nli = None
        if use_nli:
            from automated_answer_grader.pipelines.grading.nli import (
                ContradictionGuard,
            )

            self.nli = ContradictionGuard()

    def _best_match(
        self, student_units: list[str], references: list[str]
    ) -> SimilarityMatch:
        if not student_units or not references:
            raise ValueError("Cannot compare empty student or reference text lists.")

        embeddings = self.encoder.encode(
            student_units + references,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        student_matrix = embeddings[: len(student_units)]
        reference_matrix = embeddings[len(student_units) :]
        similarities = student_matrix @ reference_matrix.T
        student_index, reference_index = np.unravel_index(
            np.argmax(similarities), similarities.shape
        )
        return SimilarityMatch(
            score=float(similarities[student_index, reference_index]),
            student_unit=student_units[student_index],
            reference=references[reference_index],
        )

    def _points_fraction(self, similarity: float) -> float:
        if similarity < self.partial_threshold:
            return 0.0
        if similarity >= self.full_threshold:
            return 1.0
        return float(
            (similarity - self.partial_threshold)
            / (self.full_threshold - self.partial_threshold)
        )

    def _grade_criterion(
        self, student_units: list[str], criterion: RubricCriterion
    ) -> tuple[CriterionResult, list[str]]:
        match = self._best_match(student_units, criterion.references)
        fraction = self._points_fraction(match.score)
        contradiction_probability: float | None = None
        review = False
        reasons: list[str] = []

        # Preserve the original baseline review rules so ablation is comparable.
        if fraction == 0.0:
            review = True
            reasons.append(f"Criterion {criterion.id}: concept may be missing.")
        if abs(match.score - self.partial_threshold) <= 0.03:
            review = True
            reasons.append(f"Criterion {criterion.id}: near partial-credit threshold.")
        if abs(match.score - self.full_threshold) <= 0.03:
            review = True
            reasons.append(f"Criterion {criterion.id}: near full-credit threshold.")

        # Run NLI only when the similarity model would award positive credit.
        # This avoids extra computation for criteria already receiving zero.
        if self.nli is not None and fraction > 0:
            probabilities = self.nli.probabilities(
                premise=match.reference,
                hypothesis=match.student_unit,
            )
            contradiction_probability = probabilities["contradiction"]
            if contradiction_probability >= self.contradiction_threshold:
                review = True
                reasons.append(
                    f"Criterion {criterion.id}: possible contradiction "
                    f"(NLI={contradiction_probability:.3f})."
                )
                if self.nli_policy == "penalty":
                    fraction = 0.0

        awarded = round(criterion.points * fraction, 2)
        result = CriterionResult(
            criterion_id=criterion.id,
            description=criterion.description,
            similarity=round(match.score, 4),
            awarded_points=awarded,
            available_points=criterion.points,
            matched_student_unit=match.student_unit,
            matched_reference=match.reference,
            contradiction_probability=(
                round(contradiction_probability, 4)
                if contradiction_probability is not None
                else None
            ),
            review=review,
        )
        return result, reasons

    def grade(self, request: GradeRequest) -> GradeResponse:
        """Return score, matched evidence, and instructor-review indicators."""
        max_score = sum(criterion.points for criterion in request.criteria)
        cleaned = normalize_text(request.student_answer)
        if not cleaned:
            return GradeResponse(
                score=0.0,
                max_score=max_score,
                overall_similarity=0.0,
                review_required=True,
                review_reasons=["Student answer is empty."],
                criteria=[],
            )

        student_units = split_answer_units(cleaned)
        criterion_results: list[CriterionResult] = []
        reasons: list[str] = []
        for criterion in request.criteria:
            result, new_reasons = self._grade_criterion(student_units, criterion)
            criterion_results.append(result)
            reasons.extend(new_reasons)

        overall = self._best_match([cleaned], [request.reference_answer])
        return GradeResponse(
            score=round(sum(c.awarded_points for c in criterion_results), 2),
            max_score=round(max_score, 2),
            overall_similarity=round(overall.score, 4),
            review_required=any(c.review for c in criterion_results),
            review_reasons=reasons,
            criteria=criterion_results,
        )