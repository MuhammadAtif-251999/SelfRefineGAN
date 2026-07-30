"""PSNR, SSIM, and MAE calculations."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    prediction = np.clip(prediction, 0.0, 1.0)
    return {
        "psnr": float(
            peak_signal_noise_ratio(ground_truth, prediction, data_range=1.0)
        ),
        "ssim": float(
            structural_similarity(
                ground_truth,
                prediction,
                channel_axis=-1,
                data_range=1.0,
            )
        ),
        "mae": float(np.mean(np.abs(ground_truth - prediction))),
    }
