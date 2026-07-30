"""Complete refinement network."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from selfrefine_gan.models.refinement_block import refinement_block


def build_refinement_model(
    input_shape: tuple[int, int, int],
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="refinement_input")

    features = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
    )(inputs)
    residual_output = refinement_block(features)
    reconstructed_image = layers.Conv2D(
        3,
        (3, 3),
        activation="sigmoid",
        padding="same",
    )(residual_output)
    concatenated = layers.Concatenate()(
        [reconstructed_image, features, residual_output]
    )
    output = layers.Conv2D(
        3,
        (3, 3),
        activation="relu",
        padding="same",
        name="refined_image",
    )(concatenated)

    # The notebook computed another concatenation after this output but did not
    # connect it to the model output. It had no weights and no training effect.
    return tf.keras.Model(
        inputs=inputs,
        outputs=output,
        name="Refinement_Model",
    )
