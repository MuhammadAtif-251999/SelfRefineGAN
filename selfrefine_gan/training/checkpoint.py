"""Save generator, discriminator, and refinement checkpoints."""

from __future__ import annotations

from pathlib import Path

from selfrefine_gan.training.system import SelfRefinementGAN


def save_models(
    gan: SelfRefinementGAN,
    checkpoint_dir: str | Path,
    iteration: int,
) -> dict[str, Path]:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "generator": checkpoint_dir / f"generator_iter_{iteration}.h5",
        "discriminator": checkpoint_dir / f"discriminator_iter_{iteration}.h5",
        "refinement": checkpoint_dir
        / f"refinement_model_iter_{iteration}.h5",
    }

    gan.generator.save(paths["generator"], include_optimizer=False)
    gan.discriminator.save(paths["discriminator"], include_optimizer=False)
    gan.refinement_model.save(paths["refinement"], include_optimizer=False)
    return paths
