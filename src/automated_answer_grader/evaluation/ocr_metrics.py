from jiwer import cer, wer


def normalize_ocr_text(text: str) -> str:
    return " ".join(text.split())


def compute_ocr_metrics(
    references: list[str],
    predictions: list[str],
) -> dict[str, float]:

    if len(references) != len(predictions):
        raise ValueError("Reference and prediction counts must match.")

    if not references:
        raise ValueError("At least one sample is required for evaluation.")

    clean_references = [normalize_ocr_text(text) for text in references]

    clean_predictions = [normalize_ocr_text(text) for text in predictions]

    return {
        "cer": float(cer(clean_references, clean_predictions)),
        "wer": float(wer(clean_references, clean_predictions)),
    }
