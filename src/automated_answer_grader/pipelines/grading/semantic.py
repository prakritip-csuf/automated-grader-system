from __future__ import annotations

from dataclasses import dataclass

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

# 1. Store the best semantic match


@dataclass(frozen=True)
class SimilarityMatch:
    """Best match between student text and rubric references."""

    score: float
    student_unit: str
    reference: str


# 2. Main semantic grader


class SemanticGrader:
    # Grade short answers using embeddings and weighted rubrics.

    def __init__(
        self,
        embedding_model: str = ("sentence-transformers/all-MiniLM-L6-v2"),
        *,
        partial_threshold: float = 0.50,
        full_threshold: float = 0.72,
        contradiction_threshold: float = 0.60,
        use_nli: bool = False,
    ) -> None:

        if not (0.0 <= partial_threshold < full_threshold <= 1.0):
            raise ValueError("Thresholds must satisfy 0 <= partial < full <= 1.")

        if not 0.0 <= contradiction_threshold <= 1.0:
            raise ValueError("Contradiction threshold must be between 0 and 1.")

        # Load the pretrained sentence embedding model.
        self.encoder = SentenceTransformer(embedding_model)

        # Provisional grading thresholds.
        self.partial_threshold = partial_threshold
        self.full_threshold = full_threshold

        # Optional contradiction detection settings.
        self.contradiction_threshold = contradiction_threshold

        self.nli = None

        # Import NLI only if explicitly enabled.
        if use_nli:
            from automated_answer_grader.pipelines.grading.nli import (
                ContradictionGuard,
            )

            self.nli = ContradictionGuard()

    # 3. Calculate the best semantic match

    def _best_match(
        self,
        student_units: list[str],
        references: list[str],
    ) -> SimilarityMatch:
        """Find the highest similarity between student and rubric text."""

        if not student_units or not references:
            raise ValueError("Student units and references cannot be empty.")

        # Combine texts so the model can encode them.
        all_text = student_units + references

        embeddings = self.encoder.encode(
            all_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Separate student and reference embeddings.
        student_matrix = embeddings[: len(student_units)]

        reference_matrix = embeddings[len(student_units) :]

        # Normalized dot product equals cosine similarity.
        similarities = student_matrix @ reference_matrix.T

        # Locate the highest similarity in the matrix.
        student_index, reference_index = np.unravel_index(
            np.argmax(similarities),
            similarities.shape,
        )

        return SimilarityMatch(
            score=float(similarities[student_index, reference_index]),
            student_unit=student_units[student_index],
            reference=references[reference_index],
        )

    # 4. Convert similarity into a credit fraction

    def _points_fraction(
        self,
        similarity: float,
    ) -> float:
        # Map semantic similarity to a fraction of rubric points.

        if similarity < self.partial_threshold:
            return 0.0

        if similarity >= self.full_threshold:
            return 1.0

        # Linear interpolation for partial credit.
        width = self.full_threshold - self.partial_threshold

        fraction = (similarity - self.partial_threshold) / width

        return float(fraction)

    # 5. Evaluate one rubric criterion

    def _grade_criterion(
        self,
        student_units: list[str],
        criterion: RubricCriterion,
    ) -> tuple[CriterionResult, list[str]]:
        """Evaluate how well the answer satisfies one criterion."""

        # Find the closest student/reference sentence pair.
        match = self._best_match(
            student_units,
            criterion.references,
        )

        fraction = self._points_fraction(match.score)

        contradiction_probability = None
        review = False
        reasons: list[str] = []

        # Optional NLI contradiction detection.
        if self.nli is not None:
            probabilities = self.nli.probabilities(
                premise=match.reference,
                hypothesis=match.student_unit,
            )

            contradiction_probability = probabilities["contradiction"]

            if contradiction_probability >= self.contradiction_threshold:
                # Provisional contradiction policy.
                fraction = 0.0
                review = True

                reasons.append(f"Criterion {criterion.id}: possible contradiction detected.")

        # Flag similarities close to scoring boundaries.
        margin = 0.03

        if abs(match.score - self.partial_threshold) <= margin:
            review = True
            reasons.append(f"Criterion {criterion.id}: near partial-credit threshold.")

        if abs(match.score - self.full_threshold) <= margin:
            review = True
            reasons.append(f"Criterion {criterion.id}: near full-credit threshold.")

        # Flag criteria with no detected evidence.
        if fraction == 0.0:
            review = True

            if contradiction_probability is None or (
                contradiction_probability < self.contradiction_threshold
            ):
                reasons.append(f"Criterion {criterion.id}: required concept may be missing.")

        # Calculate weighted points.
        awarded_points = criterion.points * fraction

        result = CriterionResult(
            criterion_id=criterion.id,
            description=criterion.description,
            similarity=round(match.score, 4),
            awarded_points=round(awarded_points, 2),
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

    # 6. Grade the complete student answer

    def grade(
        self,
        request: GradeRequest,
    ) -> GradeResponse:
        # Evaluate all criteria and calculate the final score.

        # Normalize OCR or typed text.
        cleaned_answer = normalize_text(request.student_answer)

        # Calculate total available points.
        max_score = sum(criterion.points for criterion in request.criteria)

        # Handle blank answers.
        if not cleaned_answer:
            return GradeResponse(
                score=0.0,
                max_score=max_score,
                overall_similarity=0.0,
                review_required=True,
                review_reasons=["Student answer is empty."],
                criteria=[],
            )

        # Split into sentence-like units.
        student_units = split_answer_units(cleaned_answer)

        criterion_results: list[CriterionResult] = []
        review_reasons: list[str] = []

        # Evaluate each rubric criterion.
        for criterion in request.criteria:
            result, reasons = self._grade_criterion(
                student_units,
                criterion,
            )

            criterion_results.append(result)
            review_reasons.extend(reasons)

        # Compare the complete answer with the teacher reference.
        overall_match = self._best_match(
            [cleaned_answer],
            [request.reference_answer],
        )

        # Add awarded points from all criteria.
        total_score = sum(result.awarded_points for result in criterion_results)

        # Determine whether human review is recommended.
        review_required = any(result.review for result in criterion_results)

        return GradeResponse(
            score=round(total_score, 2),
            max_score=max_score,
            overall_similarity=round(
                overall_match.score,
                4,
            ),
            review_required=review_required,
            review_reasons=review_reasons,
            criteria=criterion_results,
        )
