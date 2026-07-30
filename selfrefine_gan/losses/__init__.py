"""SelfRefineGAN loss functions."""

from selfrefine_gan.losses.charbonnier import carbonnier_loss, charbonnier_loss
from selfrefine_gan.losses.generator_loss import CombinedGeneratorLoss
from selfrefine_gan.losses.refinement_loss import CombinedRefinementLoss
from selfrefine_gan.losses.texture_energy import TextureEnergyLoss

__all__ = [
    "charbonnier_loss",
    "carbonnier_loss",
    "TextureEnergyLoss",
    "CombinedGeneratorLoss",
    "CombinedRefinementLoss",
]
