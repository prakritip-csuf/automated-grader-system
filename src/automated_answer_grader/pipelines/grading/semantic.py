from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from automated_answer_grader.domain.schemas import(
    CriterionResult,
    GradeRequest,
    GradeResponse,
    RubricCriterion,
)

from automated_answer_grader.pipelines.grading.nli import ContradictionGuard
from automated_answer_grader.pipelines.grading.text import split_answer_units

@dataclass(frozen=True)
class SimilarityMatch:
    score: float
    student_unit: str
    reference: str

class SemanticGrader:

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        partial_threshold: float = 0.50,
        full_threshold: float = 0.72,
        contradiction_threshold: float = 0.60,
        use_nli: bool = False,
    ) -> None:
        if partial_threshold >= full_threshold:
            raise ValueError("partial_threshold must be below full_threshold")

        self.encoder = SentenceTransformer(embedding_model)
        self.partial_threshold = partial_threshold
        self.full_threshold = full_threshold
        self.contraction_threshold = self.contraction_threshold
        self.nli = ContradictionGuard() if use_nli else None

    def _best_match(
        self,
        student_units: list[str],
        references: list[str],
    ) -> SimilarityMatch:
        all_text = student_units + references
        embeddings = self.encoder.encode(
            all_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        student_matrix = embeddings[:len(student_units)]
        reference_matrix = embeddings[len(student_units):]

        similarities = student_matrix @ reference_matrix.T
        student_index, reference_index = np.unravel_index(
            similarities.argmax(),
            similarities.shape,
        )

        return SimilarityMatch(
            score= float(similarities[student_index, reference_index]),
            student_unit=student_units[student_index],
            reference=references[reference_index],
        )

    def _points_fraction(self, similarity: float) -> float:
        if similarity < self.partial_threshold:
            return 0.0
        if similarity >= self.full_threshold:
            return 1.0

        width = self.full_threshold - self.partial_threshold
        return (similarity - self.partial_threshold) / width

    def _grade_criterion(
        self,
        student_units: list[str],
        criterion: RubricCriterion,
    ) -> tuple[CriterionResult, list[str]]:
        match = self._best_match(student_units, criterion.references)
        fraction = self._points_fraction(match.score)
        contradiction_probability: float | None = None
        review = False
        reasons: list[str] = []

        if self.nli is not None:
            nli_probs = self.nli.probabilities(
                premise=match.reference,
                hypothesis=match.student_unit,
            )
            contradiction_probability = nli_probs["contradiction"]

            if contradiction_probability >= self.contraction_threshold:
                fraction = 0.0
                review = True
                reasons.append(
                    f"Criterion {criterion.id}: likely contradiction"
                )

        margin = 0.03
        if abs(match.score - self.partial_threshold) <= margin:
            review=True
            reasons.append(
                f"Criterion {criterion.id}: near partial-credit boundary"
            )

        if abs(match.score - self.full_threshold) <= margin:
            review= True
            reasons.append(
                f"Criterion {criterion.id}: near full-credit boundary"
            )

        points = criterion.points * fraction

        return(
            CriterionResult(
                criterion_id=criterion.id,
                description=criterion.description,
                similarity=round(match.score, 4),
                awarded_points=round(points, 2),
                available_points=criterion.points,
                matched_student_unit=match.student_unit,
                matched_reference=match.reference,
                contradiction_probability=(
                    round(contradiction_probability, 4)
                    if contradiction_probability is not None
                    else None
                ),
                review=review,
            ),
            reasons,
        )

    def grade(self, request: GradeRequest) -> GradeResponse:
        student_units = split_answer_units(request.student_answer)
        if not student_units:
            max_score = sum(item.points for item in request.criteria)
            return GradeResponse(
                score=0.0,
                max_score=max_score,
                overall_similarity=0.0,
                review_required=False,
                review_reasons=["No answer provided."],
                criteria=[],
            )

        criterion_results: list[CriterionResult] = []
        review_reasons: list[str] = []

        for criterion in request.criteria:
            result, reasons = self._grade_criterion(
                student_units,
                criterion
            )
            criterion_results.append(result)
            review_reasons.extend(reasons)

        overall_match = self._best_match(
            student_units,
            [request.reference_answer],
        )

        score = sum(item.awarded_points for item in criterion_results)
        max_score = sum(item.available_points for item in criterion_results)
        review_required = any(item.review for item in criterion_results)

        return GradeResponse(
            score=round(score, 2),
            max_score=round(max_score, 2),
            overall_similarity=round(overall_match.score,4),
            review_required=review_required,
            review_reasons=review_reasons,
            criteria=criterion_results,
        )