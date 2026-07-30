"""PIL/NumPy preprocessing, postprocessing, and model loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def preprocess_pil_image(
    image: Image.Image,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    resized = image.convert("RGB").resize((image_width, image_height))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def postprocess_prediction(
    prediction: np.ndarray,
    output_size: tuple[int, int] | None = None,
) -> Image.Image:
    output = np.clip(prediction[0] * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(output, mode="RGB")
    if output_size is not None:
        image = image.resize(output_size)
    return image


def load_keras_model(path: str | Path) -> tf.keras.Model:
    model_path = str(path)
    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        return tf.keras.models.load_model(model_path, compile=False)
