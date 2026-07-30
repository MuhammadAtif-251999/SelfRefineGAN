"""Sobel-gradient texture-energy loss."""

from __future__ import annotations

import tensorflow as tf


class TextureEnergyLoss(tf.keras.losses.Loss):
    def __init__(self, name: str = "texture_energy_loss") -> None:
        super().__init__(name=name)

    def call(
        self,
        generated_image: tf.Tensor,
        target_image: tf.Tensor,
    ) -> tf.Tensor:
        generated_gradients_x = tf.image.sobel_edges(generated_image)[..., 0]
        generated_gradients_y = tf.image.sobel_edges(generated_image)[..., 1]
        target_gradients_x = tf.image.sobel_edges(target_image)[..., 0]
        target_gradients_y = tf.image.sobel_edges(target_image)[..., 1]

        generated_energy = generated_gradients_x**2 + generated_gradients_y**2
        target_energy = target_gradients_x**2 + target_gradients_y**2
        return tf.reduce_mean(tf.square(generated_energy - target_energy))
