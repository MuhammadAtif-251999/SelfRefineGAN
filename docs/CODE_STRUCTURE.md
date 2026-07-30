# File-based source mapping

This repository contains no notebook. The original implementation has been separated as follows:

| Component | Source file |
|---|---|
| Gated residual block | `selfrefine_gan/blocks/gated_residual.py` |
| Refinement residual block | `selfrefine_gan/blocks/refinement_residual.py` |
| Generator | `selfrefine_gan/models/generator.py` |
| Refinement branch | `selfrefine_gan/models/refinement_block.py` |
| Refinement model | `selfrefine_gan/models/refinement_network.py` |
| Discriminator | `selfrefine_gan/models/discriminator.py` |
| Charbonnier loss | `selfrefine_gan/losses/charbonnier.py` |
| Texture-energy loss | `selfrefine_gan/losses/texture_energy.py` |
| Generator combined loss | `selfrefine_gan/losses/generator_loss.py` |
| Refinement combined loss | `selfrefine_gan/losses/refinement_loss.py` |
| Discriminator update | `selfrefine_gan/training/discriminator_step.py` |
| Generator/refinement update | `selfrefine_gan/training/generator_refinement_step.py` |
| Three-network container | `selfrefine_gan/training/system.py` |
| Iteration loop | `selfrefine_gan/training/trainer.py` |
| Checkpoint saving | `selfrefine_gan/training/checkpoint.py` |
| Training CSV | `selfrefine_gan/training/history.py` |
| Dataset pairing | `selfrefine_gan/data/file_pairs.py` |
| Image preprocessing | `selfrefine_gan/data/preprocessing.py` |
| TensorFlow dataset | `selfrefine_gan/data/dataset.py` |
| Inference engine | `selfrefine_gan/inference/predictor.py` |
| Metrics | `selfrefine_gan/evaluation/metrics.py` |
| Evaluation loop | `selfrefine_gan/evaluation/evaluator.py` |
| Main training command | `train.py` |
| Main inference command | `infer.py` |
| Main evaluation command | `evaluate.py` |

No model architecture, optimizer learning rate, loss coefficient, update order, dataset split, iteration count, preview interval, or checkpoint interval was intentionally changed.
