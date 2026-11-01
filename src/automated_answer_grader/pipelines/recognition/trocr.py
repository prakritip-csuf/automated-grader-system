from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from automated_answer_grader.pipelines.preprocessing.handwriting import (
    segment_lines,
)


# store complete transcription and individual lines
@dataclass(frozen=True)
class RecognitionOutput:
    text: str
    lines: list[str]


class TrOCRRecognizer:
    # recognize single-line and multi-line handwritten text
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading TrOCR model on {self.device}...")

        self.processor = TrOCRProcessor.from_pretrained(model_name)

        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        print(f"TrOCR model loaded successfully.")

    def recognize_line(self, image: Image.Image) -> str:
        image = image.convert("RGB")

        # convert image into tensors expected by TROCR
        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.to(self.device)

        with torch.inference_mode():
            # generate predicted text token IDs
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=1,
            )

        # convert token IDs into readable text
        predicted_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return predicted_text.strip()

    def recognize_answer(
        self,
        image: Image.Image,
    ) -> RecognitionOutput:
        line_images = segment_lines(image)
        print(f"Detected {len(line_images)} handwritten lines.")

        if not line_images:
            return RecognitionOutput(
                text="",
                lines=[],
            )

        recognized_lines: list[str] = []

        for index, line_image in enumerate(
            line_images,
            start=1,
        ):
            text = self.recognize_line(line_image)
            recognized_lines.append(text)

            print(f"Line {index}: {text}")

        # combine lines in top-to-bottom reading order
        complete_answer = "\n".join(recognized_lines)

        return RecognitionOutput(
            text=complete_answer,
            lines=recognized_lines,
        )
