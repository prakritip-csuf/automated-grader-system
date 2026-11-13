import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from automated_answer_grader.core.dependencies import (
    get_semantic_grader,
)
from automated_answer_grader.domain.schemas import GradeRequest


def main() -> None:
    base = Path("data/evaluation/grading")

    # Use validation first. Reserve test.csv for final evaluation.
    data = pd.read_csv(base / "validation.csv")

    rubrics = json.loads(
        (base / "rubrics.json").read_text(encoding="utf-8")
    )

    grader = get_semantic_grader()
    results = []

    for row in data.itertuples(index=False):
        rubric = rubrics[str(row.question_id)]

        request = GradeRequest(
            question=rubric["question"],
            reference_answer=rubric["reference_answer"],
            student_answer=str(row.student_answer),
            criteria=rubric["criteria"],
        )

        prediction = grader.grade(request)

        results.append({
            "response_id": row.response_id,
            "question_id": row.question_id,
            "human_score": float(row.human_score),
            "predicted_score": prediction.score,
            "max_score": prediction.max_score,
            "absolute_error": abs(
                prediction.score - float(row.human_score)
            ),
            "review_required": prediction.review_required,
        })

    df = pd.DataFrame(results)

    human = df["human_score"].to_numpy()
    predicted = df["predicted_score"].to_numpy()

    # MAE uses the original 0-10 point scale.
    mae = mean_absolute_error(human, predicted)

    # QWK requires ordinal score categories.
    human_ordinal = np.rint(human).astype(int)
    predicted_ordinal = np.rint(predicted).astype(int)

    qwk = cohen_kappa_score(
        human_ordinal,
        predicted_ordinal,
        labels=list(range(11)),
        weights="quadratic",
    )

    within_one = np.mean(
        np.abs(human - predicted) <= 1.0
    )

    output = Path("artifacts/grading")
    output.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output / "validation_predictions.csv",
        index=False,
    )

    print("\nSemantic Grading Evaluation")
    print(f"Responses evaluated: {len(df)}")
    print(f"MAE (points): {mae:.4f}")
    print(f"QWK (rounded 0-10): {qwk:.4f}")
    print(f"Within 1 point: {within_one * 100:.2f}%")
    print(
        f"Review rate: "
        f"{df['review_required'].mean() * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
