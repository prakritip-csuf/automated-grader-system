from pydantic import BaseModel, Field, field_validator, model_validator


class RubricCriterion(BaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    points: float = Field(gt=0)
    references: list[str] = Field(min_length=1)

    @field_validator("id", "description")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Field cannor be empty.")

        return value

    @field_validator("references")
    @classmethod
    def validate_references(
        cls,
        references: list[str],
    ) -> list[str]:
        cleaned = [text.strip() for text in references]

        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Rubric references must contain non-empty text.")

        return cleaned


class GradeRequest(BaseModel):
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    student_answer: str
    criteria: list[RubricCriterion] = Field(min_length=1)

    @field_validator("question", "reference_answer")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Field cannot be empty.")

        return value

    @model_validator(mode="after")
    def validate_criterion_ids(self) -> "GradeRequest":
        ids = [criterion.id for criterion in self.criteria]

        if len(ids) != len(set(ids)):
            raise ValueError("Rubric criterion IDs must be unique.")

        return self


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
