"""Random seed and TensorFlow runtime configuration."""

from __future__ import annotations

import random

import numpy as np
import tensorflow as tf


def configure_runtime(seed: int = 42, memory_growth: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU found; running on CPU.")
        return

    if memory_growth:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print(f"Could not enable GPU memory growth: {error}")

    print(f"GPU is being used ({len(gpus)} device(s)).")
