import argparse
from pathlib import Path

from PIL import Image

from automated_answer_grader.pipelines.preprocessing.handwriting import (
    segment_lines,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    args = parser.parse_args()

    image = Image.open(args.image_path).convert("RGB")
    lines = segment_lines(image)

    output_dir = Path("artifacts/segmentation_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detected {len(lines)} lines")

    for index, line in enumerate(lines, start=1):
        output_path = output_dir / f"line_{index:02d}.png"
        line.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
