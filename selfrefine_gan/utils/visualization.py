"""Training preview visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_training_preview(
    *,
    low_batch: np.ndarray,
    gt_batch: np.ndarray,
    refined_batch: np.ndarray,
    generated_batch: np.ndarray,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = [low_batch[0], gt_batch[0], refined_batch[0], generated_batch[0]]
    titles = ["Input Image", "GT Image", "Refined Image", "Generated Image"]

    figure, axes = plt.subplots(1, 4, figsize=(12, 5))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(np.clip(image, 0.0, 1.0))
        axis.axis("off")
        axis.set_title(title)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
