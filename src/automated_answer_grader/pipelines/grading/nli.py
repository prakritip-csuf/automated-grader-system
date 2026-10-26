from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

class ContradictionGuard:
    def __init__(
        self,
        model_name: str= "cross-encoder/nli-deberta-v3-small",
    ) -> None:
        self.model = CrossEncoder(model_name)
        self.labels = ["contradiction", "entailment", "neutral"]

def probabilities(
        self,
        premise: str,
        hypothesis: str,
) -> dict[str, float]:
    logits = self.model.predict([(premise, hypothesis)])[0]
    logits = np.asarray(logits, dtype=float)
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return {
        label: float(prob)
        for label, prob in zip(self.labels, prob)
    }