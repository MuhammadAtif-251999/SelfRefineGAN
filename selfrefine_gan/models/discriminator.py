"""Four-stage convolutional discriminator."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def build_discriminator(
    input_shape: tuple[int, int, int] = (256, 256, 3),
) -> tf.keras.Model:
    model = tf.keras.Sequential(name="Discriminator")
    model.add(layers.Input(shape=input_shape, name="discriminator_input"))
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model
