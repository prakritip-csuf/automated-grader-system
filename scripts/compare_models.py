"""Compare TF-IDF, MiniLM, and MPNet for rubric-aware short-answer grading.

Run from the repository root:
    uv run python -m scripts.compare_models
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from automated_answer_grader.pipelines.grading.semantic import (
    SemanticGrader,
    SimilarityMatch,
)
from automated_answer_grader.pipelines.grading.text import split_answer_units
from scripts.calibrate_thresholds import (
    load_dataset,
    metrics,
    predict_from_features,
    prepare_features,
)


MODEL_IDS = {
    "tfidf": "tfidf",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

# Broad, model-specific calibration on the same grid. These are cosine
# similarities, not probabilities. Includes MiniLM's previous 0.20 / 0.80.
PARTIAL_CANDIDATES = [round(float(x), 2) for x in np.arange(0.00, 0.71, 0.05)]
FULL_CANDIDATES = [round(float(x), 2) for x in np.arange(0.20, 1.01, 0.05)]


class TfidfGrader(SemanticGrader):
    """TF-IDF baseline with exactly the same scoring rule as SemanticGrader.

    Fit vocabulary/IDF only on calibration answers and instructor rubrics.
    Validation answers are never used to fit the vectorizer.
    """

    def __init__(self, calibration: pd.DataFrame, rubrics: dict) -> None:
        # Skip the parent constructor: TF-IDF does not need a neural model.
        self.partial_threshold = 0.20
        self.full_threshold = 0.80
        self.contradiction_threshold = 0.60
        self.nli = None
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
        )

        training_texts: list[str] = []
        for answer in calibration["student_answer"].astype(str):
            training_texts.extend(split_answer_units(answer))
        for rubric in rubrics.values():
            for criterion in rubric["criteria"]:
                training_texts.extend(criterion["references"])
            training_texts.append(rubric["reference_answer"])

        if not any(text.strip() for text in training_texts):
            raise ValueError("No text available to fit the TF-IDF vocabulary.")
        self.vectorizer.fit(training_texts)

    def _best_match(
        self, student_units: list[str], references: list[str]
    ) -> SimilarityMatch:
        if not student_units or not references:
            raise ValueError("Student units and references must not be empty.")

        student_vectors = self.vectorizer.transform(student_units)
        reference_vectors = self.vectorizer.transform(references)
        similarities = cosine_similarity(student_vectors, reference_vectors)
        student_index, reference_index = np.unravel_index(
            np.argmax(similarities), similarities.shape
        )
        return SimilarityMatch(
            score=float(similarities[student_index, reference_index]),
            student_unit=student_units[student_index],
            reference=references[reference_index],
        )


def verify_splits(calibration: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Check that the two development splits are disjoint."""
    overlap = set(calibration["response_id"]) & set(validation["response_id"])
    if overlap:
        raise ValueError(f"Response IDs overlap between splits: {len(overlap)}")

    if "student_id" in calibration and "student_id" in validation:
        calib_students = set(calibration["student_id"]) - {""}
        valid_students = set(validation["student_id"]) - {""}
        overlap = calib_students & valid_students
        if overlap:
            raise ValueError(f"Student IDs overlap between splits: {len(overlap)}")


def make_model(name: str, calibration: pd.DataFrame, rubrics: dict):
    if name == "tfidf":
        return TfidfGrader(calibration, rubrics)
    return SemanticGrader(
        embedding_model=MODEL_IDS[name],
        partial_threshold=0.20,
        full_threshold=0.80,
        use_nli=False,
    )


def calibrate_model(grader, calibration_features: list[dict]) -> tuple[pd.DataFrame, float, float]:
    """Choose thresholds by minimum calibration MAE only."""
    trials: list[dict] = []
    for partial in PARTIAL_CANDIDATES:
        for full in FULL_CANDIDATES:
            if partial >= full:
                continue
            predictions = predict_from_features(
                grader, calibration_features, partial, full
            )
            trials.append(
                {
                    "partial_threshold": partial,
                    "full_threshold": full,
                    **metrics(predictions),
                }
            )

    grid = pd.DataFrame(trials).sort_values(
        ["mae_points", "partial_threshold", "full_threshold"],
        ascending=[True, True, True],
    )
    best = grid.iloc[0]
    return grid, float(best["partial_threshold"]), float(best["full_threshold"])


def check_live_scores(
    grader, validation: pd.DataFrame, rubrics: dict, cached_predictions: pd.DataFrame
) -> None:
    """Check a few offline scores against the production grader method."""
    from automated_answer_grader.domain.schemas import GradeRequest

    for row in validation.head(3).to_dict(orient="records"):
        rubric = rubrics[str(row["question_id"])]
        request = GradeRequest(
            question=rubric["question"],
            reference_answer=rubric["reference_answer"],
            student_answer=str(row["student_answer"]),
            criteria=rubric["criteria"],
        )
        live = grader.grade(request)
        offline = cached_predictions.loc[
            cached_predictions["response_id"] == str(row["response_id"])
        ].iloc[0]
        if not np.isclose(live.score, float(offline["predicted_score"]), atol=0.011):
            raise AssertionError(
                f"Live/offline score mismatch for {row['response_id']}: "
                f"{live.score} vs {offline['predicted_score']}"
            )
        if live.review_required != bool(offline["review_required"]):
            print(
                f"WARNING: review flag differs for {row['response_id']}. "
                "Align review policy before reporting review rates."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_IDS),
        default=["tfidf", "minilm", "mpnet"],
        help="Models to evaluate (default: all three).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/evaluation/grading")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/model_comparison")
    )
    args = parser.parse_args()

    rubrics = json.loads((args.data_dir / "rubrics.json").read_text(encoding="utf-8"))
    calibration = load_dataset(args.data_dir / "calibration.csv")
    validation = load_dataset(args.data_dir / "validation.csv")
    verify_splits(calibration, validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for name in dict.fromkeys(args.models):
        print(f"\n=== {name.upper()} ===", flush=True)
        model_start = time.perf_counter()
        grader = make_model(name, calibration, rubrics)
        load_seconds = time.perf_counter() - model_start

        print(f"Preparing {len(calibration)} calibration responses...", flush=True)
        start = time.perf_counter()
        calibration_features = prepare_features(grader, calibration, rubrics)
        calibration_seconds = time.perf_counter() - start

        print(f"Preparing {len(validation)} validation responses...", flush=True)
        start = time.perf_counter()
        validation_features = prepare_features(grader, validation, rubrics)
        validation_seconds = time.perf_counter() - start

        grid, partial, full = calibrate_model(grader, calibration_features)
        grid.to_csv(args.output_dir / f"{name}_threshold_search.csv", index=False)

        calibration_predictions = predict_from_features(
            grader, calibration_features, partial, full
        )
        validation_predictions = predict_from_features(
            grader, validation_features, partial, full
        )
        validation_predictions.to_csv(
            args.output_dir / f"{name}_validation_predictions.csv", index=False
        )

        grader.partial_threshold = partial
        grader.full_threshold = full
        check_live_scores(grader, validation, rubrics, validation_predictions)

        calib_metrics = metrics(calibration_predictions)
        valid_metrics = metrics(validation_predictions)
        summary = {
            "model": name,
            "model_checkpoint": MODEL_IDS[name],
            "partial_threshold": partial,
            "full_threshold": full,
            "calibration_mae_points": calib_metrics["mae_points"],
            **{f"validation_{k}": v for k, v in valid_metrics.items()},
            "model_load_seconds": round(load_seconds, 2),
            "validation_feature_seconds": round(validation_seconds, 2),
            "validation_seconds_per_response": round(
                validation_seconds / len(validation), 4
            ),
        }
        summaries.append(summary)
        print(
            f"Selected thresholds: partial={partial:.2f}, full={full:.2f}\n"
            f"Calibration MAE: {calib_metrics['mae_points']:.4f}\n"
            f"Validation MAE: {valid_metrics['mae_points']:.4f}\n"
            f"Validation QWK: {valid_metrics['qwk_rounded_0_to_10']}\n"
            f"Within 1 point: {100 * valid_metrics['within_one_raw_point']:.1f}%\n"
            f"Review rate: {100 * valid_metrics['review_rate']:.1f}%\n"
            f"Model loading: {load_seconds:.1f}s; feature extraction: "
            f"{calibration_seconds + validation_seconds:.1f}s",
            flush=True,
        )

        del grader, calibration_features, validation_features
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    comparison = pd.DataFrame(summaries).sort_values(
        ["validation_mae_points", "validation_qwk_rounded_0_to_10"],
        ascending=[True, False], na_position="last"
    )
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)

    selected = comparison.iloc[0]
    config = {
        "model": str(selected["model_checkpoint"]),
        "partial_threshold": float(selected["partial_threshold"]),
        "full_threshold": float(selected["full_threshold"]),
        "use_nli": False,
        "selection_basis": "minimum validation MAE after calibration-only threshold tuning",
        "note": (
            "TF-IDF requires a saved, calibration-fitted vectorizer before deployment. "
            "This file is an experiment result, not an automatic API configuration."
        ),
    }
    (args.output_dir / "selected_model.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print("\n=== MODEL COMPARISON (VALIDATION ONLY) ===")
    columns = [
        "model", "partial_threshold", "full_threshold",
        "validation_mae_points", "validation_qwk_rounded_0_to_10",
        "validation_within_one_raw_point", "validation_review_rate",
        "validation_seconds_per_response",
    ]
    print(comparison[columns].to_string(index=False))
    print(f"\nSelected by validation MAE: {selected['model']}")
    print(f"Saved results in {args.output_dir}")
    print("The held-out test.csv was NOT opened.")


if __name__ == "__main__":
    main()