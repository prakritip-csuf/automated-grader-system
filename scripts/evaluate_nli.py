from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from automated_answer_grader.domain.schemas import GradeRequest
from automated_answer_grader.pipelines.grading.nli import ContradictionGuard
from automated_answer_grader.pipelines.grading.semantic import SemanticGrader
from scripts.calibrate_thresholds import load_dataset, metrics


def load_validation(data_dir: Path, max_responses: int) -> list[dict]:
    """Read human-scored GradeAgentOps validation responses and rubrics."""
    rubrics = json.loads((data_dir / "rubrics.json").read_text(encoding="utf-8"))
    frame = load_dataset(data_dir / "validation.csv")
    if max_responses > 0:
        frame = frame.head(max_responses)
    records = []
    for row in frame.to_dict(orient="records"):
        rubric = rubrics[str(row["question_id"])]
        records.append({
            "response_id": str(row["response_id"]),
            "question_id": str(row["question_id"]),
            "human_score": float(row["human_score"]),
            "request": GradeRequest(
                question=rubric["question"],
                reference_answer=rubric["reference_answer"],
                student_answer=str(row["student_answer"]),
                criteria=rubric["criteria"],
            ),
        })
    return records


def load_targeted_cases(path: Path) -> list[dict]:
    """Read manually authored contradiction/entailment examples."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for case in data:
        records.append({
            "response_id": case["case_id"],
            "question_id": case["question_id"],
            "human_score": float(case["human_score"]),
            "expected_contradiction": bool(case["expected_contradiction"]),
            "request": GradeRequest(
                question=case["question"],
                reference_answer=case["reference_answer"],
                student_answer=case["student_answer"],
                criteria=case["criteria"],
            ),
        })
    return records


def grade_records(
    grader: SemanticGrader, records: list[dict], condition: str, threshold: float
) -> pd.DataFrame:
    """Grade the same responses under one NLI policy."""
    rows = []
    start = time.perf_counter()
    for index, item in enumerate(records, start=1):
        result = grader.grade(item["request"])
        probs = [
            c.contradiction_probability
            for c in result.criteria
            if c.contradiction_probability is not None
        ]
        highest_prob = max(probs) if probs else None
        flagged = any(
            "possible contradiction" in reason.lower()
            for reason in result.review_reasons
        )
        rows.append({
            "response_id": item["response_id"],
            "question_id": item["question_id"],
            "condition": condition,
            "human_score": item["human_score"],
            "predicted_score": result.score,
            "max_score": result.max_score,
            "absolute_error": round(abs(result.score - item["human_score"]), 4),
            "review_required": result.review_required,
            "nli_contradiction_flag": flagged,
            "max_contradiction_probability": highest_prob,
            "review_reasons": " | ".join(result.review_reasons),
            **(
                {"expected_contradiction": item["expected_contradiction"]}
                if "expected_contradiction" in item
                else {}
            ),
        })
        if index % 20 == 0 or index == len(records):
            print(f"  {condition}: {index}/{len(records)}", flush=True)
    frame = pd.DataFrame(rows)
    print(f"  {condition} elapsed: {time.perf_counter() - start:.1f}s")
    return frame


def grading_summary(df: pd.DataFrame) -> dict:
    result = metrics(df)
    result["nli_flag_rate"] = float(df["nli_contradiction_flag"].mean())
    return result


def contradiction_summary(df: pd.DataFrame) -> dict:
    """Measure NLI detection against targeted, manually assigned labels."""
    expected = df["expected_contradiction"].astype(bool).to_numpy()
    detected = df["nli_contradiction_flag"].astype(bool).to_numpy()
    tp = int(np.sum(expected & detected))
    fp = int(np.sum(~expected & detected))
    tn = int(np.sum(~expected & ~detected))
    fn = int(np.sum(expected & ~detected))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "score_mae_out_of_2": float(df["absolute_error"].mean()),
        "review_rate": float(df["review_required"].mean()),
    }


def compare_conditions(
    grader: SemanticGrader,
    records: list[dict],
    output: Path,
    name: str,
    threshold: float,
    targeted: bool = False,
    guard: ContradictionGuard | None = None,
) -> dict:
    """Run off, review-only, and penalty with identical input records."""
    grader.nli = None
    off = grade_records(grader, records, "without_nli", threshold)

    if guard is None:
        guard = ContradictionGuard()
    grader.nli = guard
    grader.nli_policy = "review_only"
    review = grade_records(grader, records, "nli_review_only", threshold)

    grader.nli_policy = "penalty"
    penalty = grade_records(grader, records, "nli_penalty", threshold)

    # Review-only must not alter the numerical grades.
    if not np.allclose(off["predicted_score"], review["predicted_score"]):
        raise AssertionError("Review-only NLI changed grades unexpectedly.")
    if not np.all(
        penalty["predicted_score"].to_numpy()
        <= review["predicted_score"].to_numpy() + 1e-8
    ):
        raise AssertionError("NLI penalty increased a grade unexpectedly.")

    result = {}
    for label, df in [
        ("without_nli", off),
        ("nli_review_only", review),
        ("nli_penalty", penalty),
    ]:
        df.to_csv(output / f"{name}_{label}.csv", index=False)
        result[label] = (
            contradiction_summary(df) if targeted else grading_summary(df)
        )
    if not targeted:
        result["new_review_flags_from_nli"] = int(
            (review["review_required"] & ~off["review_required"]).sum()
        )
        result["responses_with_penalized_scores"] = int(
            (penalty["predicted_score"] < off["predicted_score"]).sum()
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/evaluation/grading"))
    parser.add_argument(
        "--cases", type=Path,
        default=Path("data/evaluation/grading/nli_grading_cases.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/robustness"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--partial", type=float, default=0.20)
    parser.add_argument("--full", type=float, default=0.80)
    parser.add_argument("--nli-threshold", type=float, default=0.60)
    parser.add_argument("--max-responses", type=int, default=0)
    parser.add_argument("--cases-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading MiniLM (or chosen embedding model) once...", flush=True)
    grader = SemanticGrader(
        embedding_model=args.model,
        partial_threshold=args.partial,
        full_threshold=args.full,
        contradiction_threshold=args.nli_threshold,
        use_nli=False,
    )

    report = {
        "embedding_model": args.model,
        "partial_threshold": args.partial,
        "full_threshold": args.full,
        "nli_model": "cross-encoder/nli-deberta-v3-small",
        "nli_contradiction_threshold": args.nli_threshold,
        "evaluation_note": "Development/validation ablation; test.csv not accessed.",
    }

    guard = ContradictionGuard()
    cases = load_targeted_cases(args.cases)
    print(f"Running {len(cases)} targeted cases...", flush=True)
    report["targeted_cases"] = compare_conditions(
        grader, cases, args.output_dir, "targeted", args.nli_threshold,
        targeted=True, guard=guard,
    )

    if not args.cases_only:
        validation = load_validation(args.data_dir, args.max_responses)
        print(f"Running {len(validation)} validation responses...", flush=True)
        report["validation"] = compare_conditions(
            grader, validation, args.output_dir, "validation", args.nli_threshold,
            guard=guard,
        )

    report_path = args.output_dir / "nli_ablation_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== MILESTONE 4 RESULTS ===")
    print(json.dumps(report, indent=2))
    print(f"Saved outputs in {args.output_dir}")


if __name__ == "__main__":
    main()