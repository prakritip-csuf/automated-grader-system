from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from PIL import Image, ImageOps

from automated_answer_grader.evaluation.ocr_metrics import compute_ocr_metrics
from automated_answer_grader.pipelines.recognition.trocr import TrOCRRecognizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path, default=Path("data/evaluation/iam/ground_truth_iam.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/ocr/iam_line_results.csv"))
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as file:
        samples = list(csv.DictReader(file))
    if not samples:
        raise ValueError("No IAM evaluation rows found")

    recognizer = TrOCRRecognizer()
    references: list[str] = []
    predictions: list[str] = []
    results: list[dict[str, str | float]] = []

    for sample in samples:
        image_path = Path(sample["image_path"])
        with Image.open(image_path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        start = time.perf_counter()
        predicted = recognizer.recognize_line(rgb)
        elapsed = time.perf_counter() - start
        reference = sample["reference_text"]
        metrics = compute_ocr_metrics([reference], [predicted])
        references.append(reference)
        predictions.append(predicted)
        results.append(
            {
                "sample_id": sample["sample_id"],
                "reference_text": reference,
                "predicted_text": predicted,
                "cer": round(metrics["cer"], 5),
                "wer": round(metrics["wer"], 5),
                "seconds": round(elapsed, 3),
            }
        )
        print(f"{sample['sample_id']}: CER={metrics['cer']:.3f}, WER={metrics['wer']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    overall = compute_ocr_metrics(references, predictions)
    print(f"Overall CER: {overall['cer']:.4f} ({overall['cer'] * 100:.2f}%)")
    print(f"Overall WER: {overall['wer']:.4f} ({overall['wer'] * 100:.2f}%)")
    print(f"Detailed results: {args.output}")


if __name__ == "__main__":
    main()
