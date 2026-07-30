"""Residual block used by the refinement network."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def residual_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Preserve the original residual-block return behavior."""
    y = layers.Conv2D(filters, (1, 1), padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, (3, 3), padding="same")(y)
    y = layers.BatchNormalization()(y)

    if x.shape[-1] != filters:
        x = layers.Conv2D(filters, (1, 1), padding="same")(x)
        y = layers.Add()([x, y])
        y = layers.Activation("relu")(y)
    return y
