"""Handwritten text recognition using TrOCR."""

from PIL import Image, ImageOps
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_NAME = "microsoft/trocr-base-handwritten"


class TrOCRRecognizer:
    """Recognize a cropped handwritten text line using TrOCR."""

    def __init__(self) -> None:
        """Load the processor and pretrained TrOCR model."""

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading TrOCR on device: {self.device}")

        self.processor = TrOCRProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model = VisionEncoderDecoderModel.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

    def recognize(self, image: Image.Image) -> str:
        """Convert one cropped handwritten line image into text."""

        # Correct image orientation if the image came from a phone.
        image = ImageOps.exif_transpose(image)

        # TrOCR expects a normal RGB image.
        image = image.convert("RGB")

        # Convert the image into tensors expected by TrOCR.
        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(self.device)

        # Run inference without calculating gradients.
        with torch.inference_mode():
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=128,
            )

        # Convert generated token IDs back into readable text.
        recognized_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return recognized_text.strip()