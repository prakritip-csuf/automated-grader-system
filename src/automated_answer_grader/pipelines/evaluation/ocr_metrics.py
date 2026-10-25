from jiwer import cer, wer

def compute_ocr_metrics(
        references:list[str],
        predictions:list[str],
) -> dict[str,float]:
    if len(references)!= len(predictions):
        raise ValueError("Reference and prediction counts must match.")

    return {
        "cer": float(cer(references, predictions)),
        "wer": float(wer(references, predictions)),
    }

