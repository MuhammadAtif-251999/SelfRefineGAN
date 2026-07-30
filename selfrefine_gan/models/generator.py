"""Gated lightweight generator architecture."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from selfrefine_gan.blocks.gated_residual import novel_residual_block_gated


def build_generator(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    encoder_input = layers.Input(shape=input_shape, name="low_light_input")

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(
        encoder_input
    )
    skip_connection1 = x

    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = novel_residual_block_gated(x, 32)
    skip_connection2 = x

    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = novel_residual_block_gated(x, 16)

    x = layers.Conv2D(32, (2, 2), activation="relu", padding="same")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = novel_residual_block_gated(x, 16)
    x = layers.Concatenate()([x, skip_connection2])

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip_connection1])

    decoded = layers.Conv2D(
        3,
        (3, 3),
        activation="sigmoid",
        padding="same",
        name="enhanced_image",
    )(x)

    return tf.keras.Model(
        inputs=encoder_input,
        outputs=decoded,
        name="Gated_Generator",
    )
