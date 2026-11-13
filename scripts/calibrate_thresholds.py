"""Calibrate MiniLM rubric thresholds without using the held-out test set"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from automated_answer_grader.domain.schemas import GradeRequest
from automated_answer_grader.pipelines.grading.semantic import SemanticGrader
from automated_answer_grader.pipelines.grading.text import split_answer_units

DATA_DIR = Path("data/evaluation/grading")
OUTPUT_DIR = Path("artifacts/calibration")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BASELINE_PARTIAL = 0.50
BASELINE_FULL = 0.72

# These are candidate *cosine similarities*, not probabilities of correctness.
PARTIAL_CANDIDATES = [round(float(x), 2) for x in np.arange(0.20, 0.71, 0.05)]
FULL_CANDIDATES = [round(float(x), 2) for x in np.arange(0.45, 0.96, 0.05)]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load response records, preserving IDs and blank answers."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(
        path,
        dtype={"response_id": str, "question_id": str, "student_id": str},
        keep_default_na=False,
    )
    required = {"response_id", "question_id", "student_answer", "human_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"No responses found in {path}")
    if df["response_id"].duplicated().any():
        raise ValueError(f"Duplicate response IDs in {path}")
    return df


def prepare_features(
    grader: SemanticGrader, df: pd.DataFrame, rubrics: dict
) -> list[dict]:
    """Compute each answer's criterion similarities only once."""
    features = []
    for row in df.to_dict(orient="records"):
        question_id = str(row["question_id"])
        if question_id not in rubrics:
            raise KeyError(f"No rubric for question {question_id}")

        rubric = rubrics[question_id]
        request = GradeRequest(
            question=rubric["question"],
            reference_answer=rubric["reference_answer"],
            student_answer=str(row["student_answer"]),
            criteria=rubric["criteria"],
        )
        units = split_answer_units(request.student_answer)
        points = [float(c.points) for c in request.criteria]
        max_score = sum(points)
        human_score = float(row["human_score"])
        if not 0 <= human_score <= max_score:
            raise ValueError(
                f"Invalid human_score={human_score} for {row['response_id']} "
                f"(rubric max={max_score})"
            )

        similarities = (
            [
                float(grader._best_match(units, c.references).score)
                for c in request.criteria
            ]
            if units
            else []
        )
        features.append(
            {
                "response_id": str(row["response_id"]),
                "question_id": question_id,
                "human_score": human_score,
                "max_score": max_score,
                "points": points,
                "similarities": similarities,
                "is_blank": not bool(units),
            }
        )
    return features


def predict_from_features(
    grader: SemanticGrader, features: list[dict], partial: float, full: float
) -> pd.DataFrame:
    """Use the production grader's exact partial-credit rule on cached similarities."""
    grader.partial_threshold = partial
    grader.full_threshold = full
    rows = []
    for item in features:
        score = 0.0
        review = item["is_blank"]
        if not item["is_blank"]:
            for similarity, available_points in zip(
                item["similarities"], item["points"], strict=True
            ):
                fraction = grader._points_fraction(similarity)
                score += round(available_points * fraction, 2)
                if (
                    fraction == 0.0
                    or abs(similarity - partial) <= 0.03
                    or abs(similarity - full) <= 0.03
                ):
                    review = True
        rows.append(
            {
                "response_id": item["response_id"],
                "question_id": item["question_id"],
                "human_score": item["human_score"],
                "predicted_score": round(score, 2),
                "max_score": item["max_score"],
                "review_required": review,
            }
        )
    return pd.DataFrame(rows)


def metrics(predictions: pd.DataFrame) -> dict:
    """Match the evaluation definitions used for GradeAgentOps (0-10 scores)."""
    human = predictions["human_score"].to_numpy(dtype=float)
    predicted = predictions["predicted_score"].to_numpy(dtype=float)
    maximum = predictions["max_score"].to_numpy(dtype=float)
    human_ordinal = np.rint(10 * human / maximum).astype(int)
    predicted_ordinal = np.rint(10 * predicted / maximum).astype(int)
    qwk = cohen_kappa_score(
        human_ordinal,
        predicted_ordinal,
        labels=list(range(11)),
        weights="quadratic",
    )
    return {
        "mae_points": float(mean_absolute_error(human, predicted)),
        "qwk_rounded_0_to_10": float(qwk) if np.isfinite(qwk) else None,
        "within_one_raw_point": float(np.mean(np.abs(human - predicted) <= 1.0)),
        "mean_signed_error": float(np.mean(predicted - human)),
        "review_rate": float(predictions["review_required"].mean()),
    }


def main() -> None:
    rubrics_path = DATA_DIR / "rubrics.json"
    if not rubrics_path.is_file():
        raise FileNotFoundError(rubrics_path)
    rubrics = json.loads(rubrics_path.read_text(encoding="utf-8"))
    calibration = load_dataset(DATA_DIR / "calibration.csv")
    validation = load_dataset(DATA_DIR / "validation.csv")

    overlapping_responses = set(calibration["response_id"]) & set(validation["response_id"])
    if overlapping_responses:
        raise ValueError("Calibration and validation share response IDs")
    if "student_id" in calibration and "student_id" in validation:
        overlap = set(calibration["student_id"]) & set(validation["student_id"])
        overlap.discard("")
        if overlap:
            raise ValueError("Calibration and validation share student IDs")

    print(f"Loading {MODEL_NAME} once...")
    grader = SemanticGrader(embedding_model=MODEL_NAME, use_nli=False)
    print(f"Computing similarities: {len(calibration)} calibration responses...")
    calibration_features = prepare_features(grader, calibration, rubrics)
    print(f"Computing similarities: {len(validation)} validation responses...")
    validation_features = prepare_features(grader, validation, rubrics)

    trials = []
    for partial in PARTIAL_CANDIDATES:
        for full in FULL_CANDIDATES:
            if partial >= full:
                continue
            predictions = predict_from_features(grader, calibration_features, partial, full)
            result = metrics(predictions)
            trials.append(
                {"partial_threshold": partial, "full_threshold": full, **result}
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_df = pd.DataFrame(trials).sort_values(
        ["mae_points", "partial_threshold", "full_threshold"]
    )
    trial_df.to_csv(OUTPUT_DIR / "threshold_search.csv", index=False)

    best = trial_df.iloc[0]
    chosen_partial = float(best["partial_threshold"])
    chosen_full = float(best["full_threshold"])

    baseline_predictions = predict_from_features(
        grader, validation_features, BASELINE_PARTIAL, BASELINE_FULL
    )
    tuned_predictions = predict_from_features(
        grader, validation_features, chosen_partial, chosen_full
    )
    baseline_metrics = metrics(baseline_predictions)
    tuned_metrics = metrics(tuned_predictions)

    baseline_predictions.to_csv(OUTPUT_DIR / "validation_baseline.csv", index=False)
    tuned_predictions.to_csv(OUTPUT_DIR / "validation_calibrated.csv", index=False)

    # Sanity-check the cached-feature predictions against the actual production method.
    grader.partial_threshold = chosen_partial
    grader.full_threshold = chosen_full
    for row in validation.head(3).to_dict(orient="records"):
        rubric = rubrics[str(row["question_id"])]
        request = GradeRequest(
            question=rubric["question"],
            reference_answer=rubric["reference_answer"],
            student_answer=str(row["student_answer"]),
            criteria=rubric["criteria"],
        )
        live_score = grader.grade(request).score
        cached_score = float(
            tuned_predictions.loc[
                tuned_predictions["response_id"] == str(row["response_id"]),
                "predicted_score",
            ].iloc[0]
        )
        if not np.isclose(live_score, cached_score, atol=0.011):
            raise AssertionError(
                f"Mismatch for {row['response_id']}: live={live_score}, cached={cached_score}"
            )

    report = {
        "model": MODEL_NAME,
        "partial_threshold": chosen_partial,
        "full_threshold": chosen_full,
        "use_nli": False,
        "selection_metric": "minimum calibration MAE (raw points)",
        "calibration_count": len(calibration),
        "validation_count": len(validation),
        "baseline_validation_metrics": baseline_metrics,
        "calibrated_validation_metrics": tuned_metrics,
    }
    (OUTPUT_DIR / "selected_thresholds.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\n=== SELECTED THRESHOLDS ===")
    print(f"Partial: {chosen_partial:.2f}; Full: {chosen_full:.2f}")
    print("\n=== ORIGINAL BASELINE ON VALIDATION ===")
    print(json.dumps(baseline_metrics, indent=2))
    print("\n=== CALIBRATED MODEL ON VALIDATION ===")
    print(json.dumps(tuned_metrics, indent=2))
    print(f"\nSaved results to: {OUTPUT_DIR}")
    print("The held-out test.csv was NOT accessed.")


if __name__ == "__main__":
    main()