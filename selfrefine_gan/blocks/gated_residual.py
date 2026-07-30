"""Gated residual block used by the lightweight generator."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def novel_residual_block_gated(
    x: tf.Tensor,
    filters: int,
    gate_reduction: int = 4,
) -> tf.Tensor:
    shortcut = x

    y = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            (1, 1),
            padding="same",
            use_bias=False,
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    gate = layers.GlobalAveragePooling2D()(x)
    gate = layers.Dense(
        max(filters // gate_reduction, 1),
        activation="relu",
    )(gate)
    gate = layers.Dense(filters, activation="sigmoid")(gate)
    gate = layers.Reshape((1, 1, filters))(gate)

    output = layers.Add()(
        [
            layers.Multiply()([gate, y]),
            layers.Multiply()(
                [layers.Lambda(lambda value: 1.0 - value)(gate), shortcut]
            ),
        ]
    )
    return layers.Activation("relu")(output)
