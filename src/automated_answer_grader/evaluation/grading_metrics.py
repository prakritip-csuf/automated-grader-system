from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

def grading_metrics(
        human_scores:list[float],
        predicted_scores:list[float],
        max_scores:list[float],
)->dict[str,float]:
    if not(
        len(human_scores)== len(predicted_scores) == len(max_scores)
    ):
        raise ValueError("All metric must have the same length.")


    human_pct = np.array(human_scores) / np.array(max_scores) * 100
    predicted_pct = np.array(predicted_scores) / np.array(max_scores) * 100

    mae = mean_absolute_error(human_pct, predicted_pct)

    # Convert normalized percentages into ordinal 0...10 bands for QWK
    
    human_levels = np.rint(human_pct / 10).astype(int)
    predicted_levels = np.rint(predicted_pct / 10).astype(int)

    qwk = cohen_kappa_score(
        human_levels,
        predicted_levels,
        weights="quadratic",
    )

    exact = np.mean(
        np.isclose(human_scores, predicted_scores, atol=1e-9)
    )
    within_one = np.mean(
        np.abs(np.array(human_scores) - np.array(predicted_scores)) <= 1
    )

    return {
        "mae_percentage_points": float(mae),
        "qwk": float(qwk),
        "exact_agreement": float(exact),
        "winthin_one_point": float(within_one),
    }
    