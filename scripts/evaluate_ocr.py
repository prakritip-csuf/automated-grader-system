from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from automated_answer_grader.evaluation.ocr_metrics import (
    compute_ocr_metrics,
)

from automated_answer_grader.pipelines.recognition.trocr import (
    TrOCRRecognizer,
)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def recognize_automatic(
    recognizer: TrOCRRecognizer,
    image_path: Path,
) -> list[str]:

    image = load_image(image_path)
    result = recognizer.recognize_answer(image)

    return result.lines


def recognize_manual(
    recognizer: TrOCRRecognizer,
    directory: Path,
) -> list[str]:

    image_paths = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if not image_paths:
        raise ValueError(f"No manually cropped lines found in {directory}")

    recognized_lines = []

    for image_path in image_paths:
        image = load_image(image_path)

        text = recognizer.recognize_line(image)

        recognized_lines.append(text)

    return recognized_lines


def evaluate(
    csv_path: Path,
    output_path: Path,
    mode: str,
) -> None:

    dataset = pd.read_csv(
        csv_path,
        dtype={"sample_id": str, "writer_id": str},
        keep_default_na=False,
    )

    required_columns = {
        "sample_id",
        "image_path",
        "reference_text",
        "writer_id",
        "expected_lines",
        "manual_line_dr",
    }

    missing = required_columns - set(dataset.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    if dataset.empty:
        raise ValueError("The evaluation dataset is empty.")

    print("Loading TrOcr Model...")

    model_start = time.perf_counter()
    recognizer = TrOCRRecognizer()
    model_load_seconds = time.perf_counter() - model_start

    print(f"Model loaded in {model_load_seconds:.2f} seconds")

    results = []
    references = []
    predictions = []

    # Process every handwritten answers
    for _, row in dataset.iterrows():
        sample_id = str(row["sample_id"])
        reference = str(row["reference_text"])

        print(f"\n Evaluating sample: {sample_id}")

        start_time = time.perf_counter()

        if mode == "automatic":
            image_path = Path(str(row["image_path"]))

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            lines = recognize_automatic(
                recognizer,
                image_path,
            )

        else:
            directory_value = str(row["manual_line_dir"])

            if not directory_value:
                raise ValueError(f"Missing manual_line_dir for {sample_id}")

            directory = Path(directory_value)

            if not directory.is_dir():
                raise FileNotFoundError(directory)

            lines = recognize_manual(
                recognizer,
                directory,
            )

        elapsed = time.perf_counter() - start_time

        # Combine the recognized lines into one answer.
        prediction = "\n".join(lines)

        # Calculate metrics for this sample
        metrics = compute_ocr_metrics(
            [reference],
            [prediction],
        )

        expected_lines = int(row["expected_lines"])
        detected_lines = len(lines)

        results.append(
            {
                "sample_id": sample_id,
                "writer_id": row["writer_id"],
                "mode": mode,
                "reference_text": reference,
                "predicted_text": prediction,
                "expected_lines": expected_lines,
                "detected_lines": detected_lines,
                "line_count_correct": (detected_lines == expected_lines),
                "cer": metrics["cer"],
                "wer": metrics["wer"],
                "inference_seconds": round(elapsed, 3),
            }
        )

        references.append(reference)
        predictions.append(prediction)

        print(f"Expected lines: {expected_lines}")
        print(f"Detected lines: {detected_lines}")
        print(f"CER: {metrics['cer']:.4f}")
        print(f"WER: {metrics['wer']:.4f}")
        print(f"Inference time: {elapsed:.2f} seconds")

    # Calculate overall metrics across all answers.
    overall = compute_ocr_metrics(
        references,
        predictions,
    )

    results_df = pd.DataFrame(results)

    line_count_accuracy = results_df["line_count_correct"].mean()

    average_latency = results_df["inference_seconds"].mean()

    # Save the detailed predictions.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # Save the aggregate results.
    summary = pd.DataFrame(
        [
            {
                "mode": mode,
                "samples": len(results_df),
                "cer": overall["cer"],
                "wer": overall["wer"],
                "line_count_accuracy": line_count_accuracy,
                "average_inference_seconds": average_latency,
                "model_load_seconds": model_load_seconds,
            }
        ]
    )

    summary_path = output_path.with_name(output_path.stem + "_summary.csv")

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 50)
    print("TrOCR EVALUATION SUMMARY")
    print("=" * 50)

    print(f"Evaluation mode: {mode}")
    print(f"Total samples: {len(results_df)}")
    print(f"CER: {overall['cer']:.4f}")
    print(f"WER: {overall['wer']:.4f}")
    print(f"Line-count accuracy: {line_count_accuracy * 100:.2f}%")
    print(f"Average inference time: {average_latency:.2f} seconds")

    print(f"\nDetailed results: {output_path}")
    print(f"Summary results: {summary_path}")


def main() -> None:
    """Read command-line arguments and start evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate TrOCR handwriting recognition.")

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/evaluation/htr/ground_truth.csv"),
    )

    parser.add_argument(
        "--mode",
        choices=["automatic", "manual"],
        default="automatic",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ocr/trocr_baseline.csv"),
    )

    args = parser.parse_args()

    evaluate(
        csv_path=args.csv,
        output_path=args.output,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
