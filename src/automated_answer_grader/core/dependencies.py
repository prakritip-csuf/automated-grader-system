from functools import lru_cache

from automated_answer_grader.pipelines.grading.semantic import SemanticGrader
from automated_answer_grader.pipelines.recognition.trocr import TrOCRRecognizer

@lru_cache(maxsize=1)
def get_trocr_recognizer() -> TrOCRRecognizer:
    return TrOCRRecognizer()

@lru_cache(maxsize=1)
def get_semantic_grader() -> SemanticGrader:
    return SemanticGrader()

