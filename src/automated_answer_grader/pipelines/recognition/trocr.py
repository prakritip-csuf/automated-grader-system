from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from tranformers import TrOCRProcessor, VisionEncoderDecoderModel

from automated_answer_grader.pipelines.preprocessing.handwriting import(
    segment_lines,
)

@dataclass(frozen=True)
class RecognitionOutput:
    text: str
    lines: list[str]

class TrOCRRecognizer:

    def __init__(
        self,
        model_name: str= "microsoft/trocr-base-handwritten",
    ) -> None:
        self.device = torch.device(
            "cuda"if torch.cuda.is_available() else "cpu"
        )

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def recognize_line(self, image: Image.Image) -> str:
        pixel_values = self.processor(
            images = image,
            return_tensors = "pt",
        ).pixel_values.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=1,
            )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        return text.strip()

    def recognize_answer(self, image: Image.Image) -> RecognitionOutput:
        line_images = segment_lines(image)
        lines = [
            self.recognize_line(line)
            for line in line_images
        ]
        lines = [line for line in lines if line]

        return RecognitionOutput(
            text="\n".join(lines),
            lines=lines,
        )