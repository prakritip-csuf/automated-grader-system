from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation/iam"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    from datasets import load_dataset

    # Use test split only. This is a line-image benchmark, NOT a multiline test.
    print("Downloading/loading Teklia/IAM-line (test split)...")
    dataset = load_dataset("Teklia/IAM-line", split="test")
    count = min(args.count, len(dataset))
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ground_truth_iam.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "image_path",
                "reference_text",
                "writer_id",
                "expected_lines",
                "manual_line_dir",
            ],
        )
        writer.writeheader()
        for index in range(count):
            sample = dataset[index]
            image_name = f"iam_test_{index:04d}.png"
            image_path = images_dir / image_name
            sample["image"].convert("RGB").save(image_path)
            writer.writerow(
                {
                    "sample_id": f"iam_test_{index:04d}",
                    "image_path": image_path.as_posix(),
                    "reference_text": sample["text"],
                    "writer_id": "not_provided_by_mirror",
                    "expected_lines": 1,
                    "manual_line_dir": "",
                }
            )
    print(f"Saved {count} line images and labels to {args.output_dir}")
    print(f"Manifest: {csv_path}")


if __name__ == "__main__":
    main()
