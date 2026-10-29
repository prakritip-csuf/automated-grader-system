from fastapi import APIRouter, Depends

from automated_answer_grader.core.dependencies import get_semantic_grader
from automated_answer_grader.domain.schemas import GradeRequest, GradeResponse
from automated_answer_grader.pipelines.grading.semantic import SemanticGrader

router = APIRouter(prefix="/grading", tags=["grading"])

@router.post("/grade", response_model = GradeResponse)
def grade_answer(
    request: GradeRequest,
    grader: SemanticGrader = Depends(get_semantic_grader),
)-> GradeResponse:
    return grader.grade(request)