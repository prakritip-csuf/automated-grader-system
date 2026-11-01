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
    padding: int = 8,
    min_line_height: int = 12,
    smooth_window: int = 5,
) -> list[Image.Image]:
    """Segment a cropped handwritten answer into text lines."""

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    # Step 1: Threshold the image.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    # Step 2: Remove small noise and large background regions.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(binary)

    for component in range(1, count):
        x, y, w, h, area = stats[component]

        if area < 8:
            continue

        # Ignore large regions such as shadows or page edges.
        if w > width * 0.85 and h > height * 0.10:
            continue

        # Ignore components touching the bottom edge if they occupy a substantial area.
        if y + h >= height - 2 and area > width * 10:
            continue

        cleaned[labels == component] = 255

    # Step 3: Compute horizontal projection.
    projection = np.count_nonzero(
        cleaned,
        axis=1,
    ).astype(np.float32)

    # Step 4: Smooth the projection.
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(
        projection,
        kernel,
        mode="same",
    )

    # Step 5: Detect regions with meaningful handwriting.
    threshold = max(2.0, float(smoothed.max()) * 0.04)
    active = smoothed > threshold

    boundaries = np.diff(
        np.pad(active.astype(np.int8), (1, 1))
    )

    starts = np.where(boundaries == 1)[0]
    ends = np.where(boundaries == -1)[0]

    # Step 6: Create line crops.
    lines = []

    for start, end in zip(starts, ends):
        if end - start < min_line_height:
            continue

        top = max(0, int(start) - padding)
        bottom = min(height, int(end) + padding)

        line = image.crop((0, top, width, bottom))
        lines.append(line)

    return lines


