
"""Analyze calibrated semantic grading results."""

import pandas as pd


def main() -> None:
    path = "artifacts/calibration/validation_calibrated.csv"

    df = pd.read_csv(path)

    # Calculate signed and absolute grading errors.
    df["signed_error"] = (
        df["predicted_score"] - df["human_score"]
    )

    df["absolute_error"] = df["signed_error"].abs()

    print("\n=== PERFORMANCE BY QUESTION ===")

    summary = (
        df.groupby("question_id")
        .agg(
            samples=("response_id", "count"),
            mae=("absolute_error", "mean"),
            bias=("signed_error", "mean"),
        )
        .round(3)
    )

    print(summary.to_string())

    print("\n=== TOP 10 LARGEST ERRORS ===")

    worst = df.nlargest(10, "absolute_error")

    print(
        worst[
            [
                "response_id",
                "question_id",
                "human_score",
                "predicted_score",
                "absolute_error",
            ]
        ].to_string(index=False)
    )

    print("\n=== GRADING DIRECTION ===")

    print(
        "Overgraded:",
        int((df["signed_error"] > 0).sum()),
    )

    print(
        "Undergraded:",
        int((df["signed_error"] < 0).sum()),
    )


if __name__ == "__main__":
    main()
