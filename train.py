"""Train SelfRefineGAN with the original iteration-based strategy."""

from __future__ import annotations

import argparse
from pathlib import Path

from selfrefine_gan.config import load_config, require
from selfrefine_gan.data.dataset import make_dataset
from selfrefine_gan.data.file_pairs import (
    discover_paired_images,
    split_training_images,
)
from selfrefine_gan.runtime import configure_runtime
from selfrefine_gan.training.system import SelfRefinementGAN
from selfrefine_gan.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    configure_runtime(
        seed=seed,
        memory_growth=bool(config.get("gpu_memory_growth", True)),
    )

    image_height = int(require(config, "image", "height"))
    image_width = int(require(config, "image", "width"))
    channels = int(config["image"].get("channels", 3))

    train_root = Path(require(config, "data", "train_root"))
    test_root = Path(require(config, "data", "test_root"))
    max_train_images = int(require(config, "data", "max_train_images"))
    batch_size = int(require(config, "data", "batch_size"))
    apply_resize = bool(config["data"].get("apply_resize", True))
    drop_remainder = bool(config["data"].get("drop_remainder", True))
    shuffle = bool(config["data"].get("shuffle", False))

    train_low, train_gt, val_low, val_gt = split_training_images(
        train_root,
        max_train_images,
    )
    test_low, test_gt = discover_paired_images(test_root)

    print(f"Train Dataset: ({len(train_low)}, {len(train_gt)})")
    print(f"Validation Dataset: ({len(val_low)}, {len(val_gt)})")
    print(f"Test Dataset: ({len(test_low)}, {len(test_gt)})")

    train_dataset = make_dataset(
        train_low,
        train_gt,
        image_height=image_height,
        image_width=image_width,
        batch_size=batch_size,
        apply_resize=apply_resize,
        drop_remainder=drop_remainder,
        shuffle=shuffle,
        seed=seed,
    )

    # Construct validation data for parity with the original implementation.
    if val_low:
        _ = make_dataset(
            val_low,
            val_gt,
            image_height=image_height,
            image_width=image_width,
            batch_size=batch_size,
            apply_resize=True,
            drop_remainder=drop_remainder,
            shuffle=False,
            seed=seed,
        )

    preview_dataset = make_dataset(
        test_low,
        test_gt,
        image_height=image_height,
        image_width=image_width,
        batch_size=1,
        apply_resize=True,
        drop_remainder=False,
        shuffle=False,
        seed=seed,
    )
    preview_batch = next(iter(preview_dataset.take(1)))

    optimizer_config = config["optimizers"]
    training_config = config["training"]
    generator_loss_config = config["losses"]["generator"]
    refinement_loss_config = config["losses"]["refinement"]

    gan = SelfRefinementGAN(
        input_shape=(image_height, image_width, channels),
        generator_learning_rate=float(optimizer_config["generator_lr"]),
        discriminator_learning_rate=float(
            optimizer_config["discriminator_lr"]
        ),
        refinement_learning_rate=float(optimizer_config["refinement_lr"]),
        beta_1=float(optimizer_config.get("beta_1", 0.9)),
        beta_2=float(optimizer_config.get("beta_2", 0.999)),
        generator_weight=float(training_config.get("generator_weight", 0.6)),
        refinement_weight=float(
            training_config.get("refinement_weight", 0.4)
        ),
        generator_loss_kwargs={
            "alpha": float(generator_loss_config["ssim_weight"]),
            "beta": float(generator_loss_config["mse_weight"]),
            "gamma": float(generator_loss_config["charbonnier_weight"]),
            "epsilon": float(
                generator_loss_config.get("charbonnier_epsilon", 0.0001)
            ),
        },
        refinement_loss_kwargs={
            "alpha": float(refinement_loss_config["ssim_weight"]),
            "beta": float(refinement_loss_config["mse_weight"]),
            "gamma": float(refinement_loss_config["texture_weight"]),
        },
    )

    output_config = config["output"]
    trainer = Trainer(
        gan,
        checkpoint_dir=output_config["checkpoint_dir"],
        visualization_dir=output_config["visualization_dir"],
        history_path=output_config["history_csv"],
        checkpoint_every=int(training_config.get("checkpoint_every", 100)),
        visualize_every=int(training_config.get("visualize_every", 10)),
    )
    trainer.fit(
        train_dataset,
        num_iterations=int(training_config.get("num_iterations", 500)),
        preview_batch=preview_batch,
    )


if __name__ == "__main__":
    main()
