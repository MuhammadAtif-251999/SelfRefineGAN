"""Container for the three networks, losses, and optimizers."""

from __future__ import annotations

import tensorflow as tf

from selfrefine_gan.losses.generator_loss import CombinedGeneratorLoss
from selfrefine_gan.losses.refinement_loss import CombinedRefinementLoss
from selfrefine_gan.models.discriminator import build_discriminator
from selfrefine_gan.models.generator import build_generator
from selfrefine_gan.models.refinement_network import build_refinement_model
from selfrefine_gan.training.discriminator_step import train_discriminator_step
from selfrefine_gan.training.generator_refinement_step import (
    train_generator_refinement_step,
)


class SelfRefinementGAN:
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        *,
        generator_learning_rate: float = 0.0002,
        discriminator_learning_rate: float = 0.0009,
        refinement_learning_rate: float = 0.0003,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        generator_weight: float = 0.6,
        refinement_weight: float = 0.4,
        generator_loss_kwargs: dict | None = None,
        refinement_loss_kwargs: dict | None = None,
    ) -> None:
        self.generator = build_generator(input_shape)
        self.discriminator = build_discriminator(input_shape)
        self.refinement_model = build_refinement_model(input_shape)

        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=generator_learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
        )
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=discriminator_learning_rate,
            beta_1=beta_1,
        )
        self.refinement_optimizer = tf.keras.optimizers.Adam(
            learning_rate=refinement_learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
        )

        self.generator_loss_function = CombinedGeneratorLoss(
            **(generator_loss_kwargs or {})
        )
        self.refinement_loss_function = CombinedRefinementLoss(
            **(refinement_loss_kwargs or {})
        )
        self.generator_weight = generator_weight
        self.refinement_weight = refinement_weight

    def train_discriminator(
        self,
        real_images: tf.Tensor,
        generated_images: tf.Tensor,
    ) -> tf.Tensor:
        return train_discriminator_step(
            discriminator=self.discriminator,
            optimizer=self.discriminator_optimizer,
            real_images=real_images,
            generated_images=generated_images,
        )

    def train_generator(
        self,
        low_light_images: tf.Tensor,
        ground_truth_images: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        return train_generator_refinement_step(
            generator=self.generator,
            discriminator=self.discriminator,
            refinement_model=self.refinement_model,
            generator_optimizer=self.generator_optimizer,
            refinement_optimizer=self.refinement_optimizer,
            generator_loss_function=self.generator_loss_function,
            refinement_loss_function=self.refinement_loss_function,
            low_light_images=low_light_images,
            ground_truth_images=ground_truth_images,
            generator_weight=self.generator_weight,
            refinement_weight=self.refinement_weight,
        )


# Original notebook class name retained as an alias.
Self_Refinement_GAN = SelfRefinementGAN
