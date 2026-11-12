from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error


def grading_metrics(
    human_scores: list[float],
    predicted_scores: list[float],
    max_scores: list[float],
) -> dict[str, float | None]:

    if not human_scores or not (
        len(human_scores)
        == len(predicted_scores)
        == len(max_scores)
    ):
        raise ValueError("Metric inputs must be nonempty and equal-length.")

    human = np.asarray(human_scores, dtype=float)
    predicted = np.asarray(predicted_scores, dtype=float)
    maximum = np.asarray(max_scores, dtype=float)

    if np.any(maximum <= 0):
        raise ValueError("Maximum scores must be positive.")

    if np.any(human < 0) or np.any(human > maximum):
        raise ValueError("Human scores must be within rubric limits.")

    human_pct = 100 * human / maximum
    predicted_pct = 100 * predicted / maximum

    # Normalized mean absolute error in percentage points.
    mae = mean_absolute_error(human_pct, predicted_pct)

    # Use 11 ordered categories: 0, 10, ..., 100 percent.
    human_levels = np.clip(np.rint(human_pct / 10), 0, 10).astype(int)
    predicted_levels = np.clip(np.rint(predicted_pct / 10), 0, 10).astype(int)

    qwk = cohen_kappa_score(
        human_levels,
        predicted_levels,
        labels=list(range(11)),
        weights="quadratic",
    )

    # Exact agreement uses the same normalized 10% bands as QWK.
    exact = np.mean(human_levels == predicted_levels)
    within_one_band = np.mean(
        np.abs(human_levels - predicted_levels) <= 1
    )

    return {
        "mae_percentage_points": float(mae),
        "qwk": float(qwk) if np.isfinite(qwk) else None,
        "exact_band_agreement": float(exact),
        "within_one_band_agreement": float(within_one_band),
    }
