from __future__ import annotations

from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except Exception:  # Keep placeholder demo mode usable without TensorFlow.
    tf = None

IMAGE_SIZE = 300


def open_uploaded_image(uploaded_file) -> Image.Image:
    uploaded_file.seek(0)
    image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
    uploaded_file.seek(0)
    return image


def enhance_image(image: Image.Image) -> Image.Image:
    """Apply CLAHE contrast enhancement and return a new PIL image."""
    arr = np.asarray(image.convert("RGB")).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


def preprocess_image_for_model(image: Image.Image, image_size: int = IMAGE_SIZE) -> np.ndarray:
    arr = np.asarray(image.convert("RGB")).astype(np.uint8)
    if tf is not None:
        arr = tf.image.resize(arr, [image_size, image_size]).numpy()
    else:
        arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return np.expand_dims(arr.astype("float32"), axis=0)


def build_metadata_row(request_data: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    """Build one metadata row using the schema produced by training.

    Match PAD-UFES-20 categories and the fitted imputer's np.nan convention.
    """
    aliases = {
        "gender": "sex",
        "sex": "sex",
        "age": "age",
        "smoke": "smoke",
        "drink": "drink",
        "anatom_site": "anatom_site",
        "region": "anatom_site",
        "body_site": "anatom_site",
    }
    row = {}
    lowered_request = {str(k).lower(): v for k, v in request_data.items()}
    for feature in feature_columns:
        key = str(feature).lower()
        source_key = aliases.get(key, key)
        value = lowered_request.get(source_key)
        if value is None or (isinstance(value, str) and not value.strip()):
            value = np.nan
        elif key in {"gender", "sex", "region", "anatom_site", "body_site"}:
            value = str(value).strip().upper()
        elif key in {"smoke", "drink"}:
            value = {"true": True, "false": False}.get(str(value).strip().lower(), np.nan)
        row[feature] = value
    return row
