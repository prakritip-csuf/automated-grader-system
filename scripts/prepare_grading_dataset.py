"""Convert GradeAgentOps data into our grading format."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


SOURCE = Path("data/external/gradeagentops")
OUTPUT = Path("data/evaluation/grading")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load questions, answers, and rubrics.
    records = json.loads(
        (SOURCE / "answers.json").read_text(encoding="utf-8")
    )

    # Load the two independent human grading files.
    grader_1 = pd.read_csv(
        SOURCE / "human_1.csv",
        encoding="utf-8-sig",
    )

    grader_2 = pd.read_csv(
        SOURCE / "human_2.csv",
        encoding="utf-8-sig",
    )

    keys = ["student_id", "question_id"]

    if (
        grader_1.duplicated(keys).any()
        or grader_2.duplicated(keys).any()
    ):
        raise ValueError("Duplicate human grading records found.")

    scores = grader_1[
        keys + ["human_final_score"]
    ].merge(
        grader_2[keys + ["human_final_score"]],
        on=keys,
        suffixes=("_1", "_2"),
        validate="one_to_one",
    )

    scores["human_score"] = (
        scores["human_final_score_1"]
        + scores["human_final_score_2"]
    ) / 2.0

    score_lookup = {
        (row.student_id, row.question_id): row.human_score
        for row in scores.itertuples(index=False)
    }

    rubrics = {}
    responses = []

    selected_questions = {
        "Q1", "Q2", "Q3", "Q4", "Q5"
    }

    for record in records:
        qid = str(record["question_id"])
        sid = str(record["student_id"])

        if qid not in selected_questions:
            continue

        key = (sid, qid)

        if key not in score_lookup:
            raise ValueError(f"Missing human grade for {key}")

        if qid not in rubrics:
            criteria = []

            for index, point in enumerate(
                record["Gold_points"], start=1
            ):
                description = point["point"]

                criteria.append({
                    "id": f"{qid}_c{index}",
                    "description": description,
                    "points": float(point["weight"]),
                    "references": [description],
                })

            rubrics[qid] = {
                "question": record["question"],
                "reference_answer": record["Gold_answer"],
                "criteria": criteria,
            }

        responses.append({
            "response_id": record["answer_id"],
            "student_id": sid,
            "question_id": qid,
            "student_answer": record["answer"],
            "human_score": float(score_lookup[key]),
        })

    data = pd.DataFrame(responses)

    # Save the converted rubrics and responses.
    (OUTPUT / "rubrics.json").write_text(
        json.dumps(rubrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    data.to_csv(OUTPUT / "responses.csv", index=False)

    # Split by student to avoid student overlap.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    development_idx, test_idx = next(
        splitter.split(data, groups=data["student_id"])
    )

    development = data.iloc[development_idx].reset_index(
        drop=True
    )
    test = data.iloc[test_idx]

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    calibration_idx, validation_idx = next(
        splitter_2.split(
            development,
            groups=development["student_id"],
        )
    )

    calibration = development.iloc[calibration_idx]
    validation = development.iloc[validation_idx]

    calibration.to_csv(
        OUTPUT / "calibration.csv", index=False
    )
    validation.to_csv(
        OUTPUT / "validation.csv", index=False
    )
    test.to_csv(
        OUTPUT / "test.csv", index=False
    )

    print("Dataset preparation completed.")
    print(f"Calibration: {len(calibration)}")
    print(f"Validation: {len(validation)}")
    print(f"Test: {len(test)}")
    print(f"Questions: {len(rubrics)}")


if __name__ == "__main__":
    main()
