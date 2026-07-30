"""Joint generator and refinement optimization step."""

from __future__ import annotations

import tensorflow as tf


def train_generator_refinement_step(
    *,
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    refinement_model: tf.keras.Model,
    generator_optimizer: tf.keras.optimizers.Optimizer,
    refinement_optimizer: tf.keras.optimizers.Optimizer,
    generator_loss_function: tf.keras.losses.Loss,
    refinement_loss_function: tf.keras.losses.Loss,
    low_light_images: tf.Tensor,
    ground_truth_images: tf.Tensor,
    generator_weight: float,
    refinement_weight: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as generator_tape, tf.GradientTape() as refinement_tape:
        generated_images = generator(low_light_images, training=True)

        # Preserved from the original code. This discriminator output is
        # evaluated during the generator step but is not part of the published
        # generator loss expression.
        _ = discriminator(generated_images, training=True)
        generator_loss = generator_loss_function(
            ground_truth_images,
            generated_images,
        )

        refined_images = refinement_model(low_light_images, training=True)
        _ = discriminator(refined_images, training=True)
        refinement_loss = refinement_loss_function(
            ground_truth_images,
            refined_images,
        )

        total_loss = (
            refinement_weight * refinement_loss
            + generator_weight * generator_loss
        )

    refinement_gradients = refinement_tape.gradient(
        refinement_loss,
        refinement_model.trainable_variables,
    )
    refinement_optimizer.apply_gradients(
        zip(refinement_gradients, refinement_model.trainable_variables)
    )

    generator_gradients = generator_tape.gradient(
        total_loss,
        generator.trainable_variables,
    )
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    return total_loss, refinement_loss
