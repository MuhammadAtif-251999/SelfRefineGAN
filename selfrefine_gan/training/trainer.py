"""Iteration-based training loop matching the original strategy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from selfrefine_gan.training.checkpoint import save_models
from selfrefine_gan.training.history import append_history
from selfrefine_gan.training.system import SelfRefinementGAN
from selfrefine_gan.utils.visualization import save_training_preview


class Trainer:
    def __init__(
        self,
        gan: SelfRefinementGAN,
        *,
        checkpoint_dir: str | Path,
        visualization_dir: str | Path,
        history_path: str | Path,
        checkpoint_every: int = 100,
        visualize_every: int = 10,
    ) -> None:
        self.gan = gan
        self.checkpoint_dir = Path(checkpoint_dir)
        self.visualization_dir = Path(visualization_dir)
        self.history_path = Path(history_path)
        self.checkpoint_every = checkpoint_every
        self.visualize_every = visualize_every

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        num_iterations: int,
        preview_batch: tuple[tf.Tensor, tf.Tensor] | None = None,
    ) -> None:
        """Run one full dataset pass for each iteration."""
        for iteration in range(1, num_iterations + 1):
            last_generator_loss: tf.Tensor | None = None
            last_refinement_loss: tf.Tensor | None = None
            last_discriminator_loss: tf.Tensor | None = None
            batch_count = 0

            for low_light_images, ground_truth_images in train_dataset:
                # Equivalent to the notebook's generator.predict call.
                generated_for_discriminator = self.gan.generator.predict(
                    low_light_images,
                    verbose=0,
                )

                last_discriminator_loss = self.gan.train_discriminator(
                    ground_truth_images,
                    generated_for_discriminator,
                )
                (
                    last_generator_loss,
                    last_refinement_loss,
                ) = self.gan.train_generator(
                    low_light_images,
                    ground_truth_images,
                )
                batch_count += 1

            if batch_count == 0:
                raise RuntimeError(
                    "The training dataset produced zero batches. Check the "
                    "dataset paths, batch size, and drop_remainder setting."
                )

            assert last_generator_loss is not None
            assert last_refinement_loss is not None
            assert last_discriminator_loss is not None

            generator_value = float(tf.reduce_mean(last_generator_loss).numpy())
            refinement_value = float(tf.reduce_mean(last_refinement_loss).numpy())
            discriminator_value = float(
                tf.reduce_mean(last_discriminator_loss).numpy()
            )

            # The original loop reported the final batch from each iteration.
            print(
                f"Iteration {iteration}/{num_iterations} | "
                f"Generator Loss: {generator_value:.6f} | "
                f"Refinement Loss: {refinement_value:.6f} | "
                f"Discriminator Loss: {discriminator_value:.6f}"
            )

            append_history(
                self.history_path,
                {
                    "iteration": iteration,
                    "generator_loss": generator_value,
                    "refinement_loss": refinement_value,
                    "discriminator_loss": discriminator_value,
                },
            )

            if (
                preview_batch is not None
                and self.visualize_every > 0
                and iteration % self.visualize_every == 0
            ):
                self._save_preview(iteration, preview_batch)

            if (
                self.checkpoint_every > 0
                and iteration % self.checkpoint_every == 0
            ):
                paths = save_models(
                    self.gan,
                    self.checkpoint_dir,
                    iteration,
                )
                print(
                    f"Saved checkpoint at iteration {iteration}: "
                    f"{paths['generator'].parent}"
                )

    def _save_preview(
        self,
        iteration: int,
        preview_batch: tuple[tf.Tensor, tf.Tensor],
    ) -> None:
        low_images, gt_images = preview_batch
        refined = self.gan.refinement_model.predict(low_images, verbose=0)
        generated = self.gan.generator.predict(low_images, verbose=0)
        save_training_preview(
            low_batch=np.asarray(low_images),
            gt_batch=np.asarray(gt_images),
            refined_batch=refined,
            generated_batch=generated,
            output_path=self.visualization_dir
            / f"iteration_{iteration:04d}.png",
        )
