from pydantic import BaseModel, Field

class RubricCriterion(BaseModel):
    id: str
    description: str
    points: float = Field(gt=0)
    references: list[str] = Field(min_length=1)

class GradeRequest(BaseModel):
    question: str
    reference_answer: str
    student_answer: str
    criteria: list[RubricCriterion] = Field(min_length=1)

class CriterionResult(BaseModel):
    criterion_id: str
    description: str
    similarity: float
    awarded_points: float
    available_points: float
    matched_student_unit: str
    matched_reference: str
    contradiction_probability: float | None = None
    review: bool = False

class GradeResponse(BaseModel):
    score: float
    max_score: float
    overall_similarity: float
    review_required: bool
    review_reasons: list[str]
    criteria: list[CriterionResult]