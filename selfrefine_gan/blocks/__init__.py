"""Reusable model blocks."""

from selfrefine_gan.blocks.gated_residual import novel_residual_block_gated
from selfrefine_gan.blocks.refinement_residual import residual_block

__all__ = ["novel_residual_block_gated", "residual_block"]
