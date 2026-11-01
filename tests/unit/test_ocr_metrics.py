import pytest

from automated_answer_grader.evaluation.ocr_metrics import (
    compute_ocr_metrics,
    normalize_ocr_text,
)


def test_perfect_recognition() -> None:
    result = compute_ocr_metrics(
        ["The quick brown fox"],
        ["The quick brown fox"],
    )

    assert result["cer"] == 0.0
    assert result["wer"] == 0.0


def test_character_substitution() -> None:
    result = compute_ocr_metrics(["cat"], ["bat"])

    assert result["cer"] == pytest.approx(1 / 3)
    assert result["wer"] == pytest.approx(1.0)


def test_whitespace_normalization() -> None:
    result = normalize_ocr_text("The quick\n brown    fox")

    assert result == "The quick brown fox"


def test_mismatched_sample_counts() -> None:
    with pytest.raises(ValueError):
        compute_ocr_metrics(
            ["Reference one", "Reference two"],
            ["Prediction one"],
        )
