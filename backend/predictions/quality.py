from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def analyze_image_quality(image: Image.Image) -> dict:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Laplacian variance is the standard sharpness metric in computer vision.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings = []
    if min(width, height) < 224:
        warnings.append("The image resolution is small. A clearer image may improve reliability.")
    if brightness < 45:
        warnings.append("The image appears dark. Prediction confidence may be affected.")
    if brightness > 220:
        warnings.append("The image appears very bright. Prediction confidence may be affected.")
    if contrast < 18:
        warnings.append("The image has low contrast. Lesion boundaries may be hard to detect.")
    if sharpness < 100:
        warnings.append("The image may be blurry. A sharper image is recommended.")

    return {
        "width": int(width),
        "height": int(height),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "warnings": warnings,
    }
