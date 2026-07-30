"""Combined generator objective: SSIM, MSE, and Charbonnier."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K

from selfrefine_gan.losses.charbonnier import charbonnier_loss


class CombinedGeneratorLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.4,
        epsilon: float = 0.0001,
        name: str = "combined_generator_loss",
    ) -> None:
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def call(
        self,
        generated_image: tf.Tensor,
        target_image: tf.Tensor,
    ) -> tf.Tensor:
        ssim_loss = 1.0 - K.mean(
            tf.image.ssim(target_image, generated_image, max_val=1.0)
        )
        mse_loss = tf.reduce_mean(tf.square(target_image - generated_image))
        charbonnier = charbonnier_loss(
            target_image,
            generated_image,
            epsilon=self.epsilon,
        )
        return (
            self.alpha * ssim_loss
            + self.beta * mse_loss
            + self.gamma * charbonnier
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
            }
        )
        return config
