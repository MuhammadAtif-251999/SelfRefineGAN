"""Combined refinement objective: SSIM, MSE, and texture energy."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K

from selfrefine_gan.losses.texture_energy import TextureEnergyLoss


class CombinedRefinementLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.7,
        name: str = "combined_refinement_loss",
    ) -> None:
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.texture_energy_loss = TextureEnergyLoss()

    def call(
        self,
        generated_image: tf.Tensor,
        target_image: tf.Tensor,
    ) -> tf.Tensor:
        ssim_loss = 1.0 - K.mean(
            tf.image.ssim(target_image, generated_image, max_val=1.0)
        )
        mse_loss = tf.reduce_mean(tf.square(target_image - generated_image))
        energy_loss = self.texture_energy_loss(generated_image, target_image)
        return (
            self.alpha * ssim_loss
            + self.beta * mse_loss
            + self.gamma * energy_loss
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}
        )
        return config
