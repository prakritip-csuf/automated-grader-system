from functools import lru_cache

from automated_answer_grader.pipelines.recognition.trocr import TrOCRRecognizer

from automated_answer_grader.pipelines.grading.semantic import SemanticGrader


@lru_cache(maxsize=1)
def get_trocr_recognizer() -> TrOCRRecognizer:
    return TrOCRRecognizer()


@lru_cache(maxsize=1)
def get_semantic_grader() -> SemanticGrader:
    return SemanticGrader(
        embedding_model=("sentence-transformers/all-MiniLM-L6-v2"),
        partial_threshold=0.50,
        full_threshold=0.72,
        use_nli=False,
    )
