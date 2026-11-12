import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from automated_answer_grader.pipelines.grading.semantic import (
    SemanticGrader,
    SimilarityMatch,
)

class TfidfGrader(SemanticGrader):
    def __init__(
            self,
            partial_threshold: float = 0.10,
            full_threshold: float = 0.40,
    ) -> None:
        if not 0 <= partial_threshold < full_threshold <= 1:
            raise ValueError("Invalid TF-IDF thresholds.")

        self.partial_threshold = partial_threshold
        self.full_threshold = full_threshold
        self.contradiction_threshold = 0.60
        self.nli = None

    def _best_match(
            self,
            student_units: list[str],
            references: list[str],
    ) -> SimilarityMatch:
        if not student_units or not references:
            raise ValueError("Texts cannot be empty.")

        texts = student_units + references

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
        )

        try:
            vectors = vectorizer.fit_transform(texts)
        except ValueError:
            return SimilarityMatch(
                score=0.0,
                student_unit=student_units[0],
                reference=references[0],
            )

        student_vectors = vectors[: len(student_units)]
        reference_vectors = vectors[len(student_units) :]

        similarities = cosine_similarity(
            student_vectors,
            reference_vectors,
        )

        i, j = np.unravel_index(
            similarities.argmax(),
            similarities.shape,
        )

        return SimilarityMatch(
            score=float(similarities[i, j]),
            student_unit=student_units[i],
            reference=references[j],
        )