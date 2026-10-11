from __future__ import annotations

import io
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(image).convert("RGB")


def segment_lines(
    image: Image.Image,
    *,
    min_area: int = 400,
    horizontal_kernel: int = 35,
    padding: int = 8,
) -> list[Image.Image]:
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel, 3))
    connected = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        connected,
        cv2.RETER_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes: list[tuple[int, int, int, int]] = []
    height, width = gray.shape

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        if w < 30 or h < 8:
            continue

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda box: box[1])

    if not boxes:
        return [image]

    return [image.crop(box) for box in boxes]
