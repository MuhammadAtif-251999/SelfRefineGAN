"""Dual-branch self-refinement block."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from selfrefine_gan.blocks.refinement_residual import residual_block


def refinement_block(inputs: tf.Tensor) -> tf.Tensor:
    x1 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(
        inputs
    )
    skip_connection1 = x1
    x1 = layers.MaxPooling2D(
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(x1)
    x1 = residual_block(x1, 128)
    x1 = layers.MaxPooling2D(
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(x1)
    x1 = residual_block(x1, 128)

    x2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(
        inputs
    )
    skip_connection2 = x2
    x2 = layers.MaxPooling2D((2, 2), padding="same")(x2)
    x2 = residual_block(x2, 64)
    x2 = layers.MaxPooling2D((2, 2), padding="same")(x2)
    x2 = residual_block(x2, 64)

    x = layers.Concatenate()([x1, x2])
    x = layers.Conv2D(64, (2, 2), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = residual_block(x, 64)
    x = layers.UpSampling2D((2, 2))(x)
    x = residual_block(x, 64)
    x = layers.Concatenate()([x, skip_connection1, skip_connection2])
    return layers.Conv2D(
        3,
        (3, 3),
        activation="sigmoid",
        padding="same",
        name="refinement_block_output",
    )(x)
