import pandas as pd


def main() -> None:
    path = "artifacts/grading/validation_predictions.csv"

    df = pd.read_csv(path)

    df["signed_error"] = (
        df["predicted_score"] - df["human_score"]
    )

    df["absolute_error"] = df["signed_error"].abs()

    print("\n--- SCORE DISTRIBUTION ---")
    print(
        df[["human_score", "predicted_score"]]
        .describe()
        .round(3)
    )

    print("\n--- GRADING BIAS ---")
    print(
        "Mean signed error:",
        round(df["signed_error"].mean(), 3),
    )

    print(
        "Overgraded responses:",
        int((df["signed_error"] > 0).sum()),
    )

    print(
        "Undergraded responses:",
        int((df["signed_error"] < 0).sum()),
    )

    print("\n--- PERFORMANCE BY QUESTION ---")

    question_results = (
        df.groupby("question_id")
        .agg(
            samples=("response_id", "count"),
            mae=("absolute_error", "mean"),
            mean_human=("human_score", "mean"),
            mean_predicted=("predicted_score", "mean"),
            review_rate=("review_required", "mean"),
        )
        .round(3)
    )

    print(question_results)

    print("\n--- TOP 10 LARGEST ERRORS ---")

    worst = df.sort_values(
        "absolute_error",
        ascending=False,
    ).head(10)

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


if __name__ == "__main__":
    main()
