"""Evaluate a generator on a paired low-light dataset."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from selfrefine_gan.data.file_pairs import discover_paired_images
from selfrefine_gan.data.preprocessing import read_image
from selfrefine_gan.evaluation.metrics import calculate_metrics
from selfrefine_gan.utils.image_io import load_keras_model


def _load_resized(path: str, height: int, width: int) -> np.ndarray:
    image = read_image(tf.constant(path))
    image = tf.image.resize(image, [height, width])
    return image.numpy().astype(np.float32)


def evaluate_dataset(
    *,
    model_path: str | Path,
    data_root: str | Path,
    image_height: int,
    image_width: int,
    output_csv: str | Path,
) -> tuple[list[dict[str, float | str]], int]:
    model = load_keras_model(model_path)
    low_paths, gt_paths = discover_paired_images(data_root)

    rows: list[dict[str, float | str]] = []
    for low_path, gt_path in zip(low_paths, gt_paths):
        low = _load_resized(low_path, image_height, image_width)
        gt = _load_resized(gt_path, image_height, image_width)

        start = time.perf_counter()
        prediction = model.predict(np.expand_dims(low, axis=0), verbose=0)[0]
        runtime = time.perf_counter() - start

        row: dict[str, float | str] = {
            "image": Path(low_path).name,
            **calculate_metrics(gt, prediction),
            "runtime_seconds": runtime,
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No images were evaluated.")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows, model.count_params()
