"""Charbonnier reconstruction loss."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K


def charbonnier_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    epsilon: float = 0.0001,
) -> tf.Tensor:
    error = K.abs(y_true - y_pred)
    return K.mean(K.sqrt(error**2 + epsilon**2))


# Preserve the spelling used in the original notebook.
carbonnier_loss = charbonnier_loss
