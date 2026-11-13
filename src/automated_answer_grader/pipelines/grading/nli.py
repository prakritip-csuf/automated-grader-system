from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ContradictionGuard:
    """Return NLI probabilities for an instructor reference and student claim.

    The reference is the premise; the student statement is the hypothesis.
    Probabilities are model outputs, NOT calibrated confidence estimates.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        max_length: int = 256,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.labels = {
            int(index): str(label).lower()
            for index, label in self.model.config.id2label.items()
        }
        expected = {"contradiction", "entailment", "neutral"}
        if set(self.labels.values()) != expected:
            raise ValueError(
                f"Unexpected NLI label mapping: {self.labels}. "
                "Check the model configuration before interpreting scores."
            )

    @lru_cache(maxsize=4096)
    def probabilities(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Return probabilities for contradiction, entailment and neutral."""
        if not premise.strip() or not hypothesis.strip():
            raise ValueError("NLI requires two nonempty statements.")

        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            scores = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        return {self.labels[index]: float(score) for index, score in enumerate(scores)}
