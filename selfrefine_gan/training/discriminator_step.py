"""One discriminator optimization step."""

from __future__ import annotations

import tensorflow as tf


def train_discriminator_step(
    discriminator: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    real_images: tf.Tensor,
    generated_images: tf.Tensor,
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        discriminator_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(real_output),
            real_output,
        ) + tf.keras.losses.binary_crossentropy(
            tf.zeros_like(fake_output),
            fake_output,
        )

    gradients = tape.gradient(
        discriminator_loss,
        discriminator.trainable_variables,
    )
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    return discriminator_loss
