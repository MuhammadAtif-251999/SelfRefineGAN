# SelfRefineGAN

Official **file-based TensorFlow/Keras implementation** of SelfRefineGAN for low-light image enhancement.

> This release does not use or include a Jupyter notebook. Every model component, loss, data operation, training step, inference operation, and evaluation operation is implemented in a separate Python source file.

The project preserves the effective architecture and training strategy of the original implementation:

- Gated lightweight generator
- Separate refinement network
- Four-stage discriminator
- Three independent Adam optimizers
- Generator loss: SSIM + MSE + Charbonnier
- Refinement loss: SSIM + MSE + texture-energy loss
- Total loss: `0.6 × generator loss + 0.4 × refinement loss`
- One complete pass through the training dataset per training iteration
- Visualization every 10 iterations
- All three models saved every 100 iterations

## Project structure

```text
SelfRefineGAN/
├── configs/
│   └── default.yaml
├── selfrefine_gan/
│   ├── blocks/
│   │   ├── gated_residual.py
│   │   └── refinement_residual.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── file_pairs.py
│   │   └── preprocessing.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── inference/
│   │   └── predictor.py
│   ├── losses/
│   │   ├── charbonnier.py
│   │   ├── generator_loss.py
│   │   ├── refinement_loss.py
│   │   └── texture_energy.py
│   ├── models/
│   │   ├── discriminator.py
│   │   ├── generator.py
│   │   ├── refinement_block.py
│   │   └── refinement_network.py
│   ├── training/
│   │   ├── checkpoint.py
│   │   ├── discriminator_step.py
│   │   ├── generator_refinement_step.py
│   │   ├── history.py
│   │   ├── system.py
│   │   └── trainer.py
│   ├── utils/
│   │   ├── image_io.py
│   │   └── visualization.py
│   ├── config.py
│   └── runtime.py
├── checkpoints/
├── outputs/
├── train.py
├── infer.py
├── evaluate.py
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux or WSL:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset structure

The default settings follow LOLv1:

```text
LOLv1/
├── our485/
│   ├── input/
│   └── gt/
└── eval15/
    ├── input/
    └── gt/
```

The low-light and ground-truth folders must contain matching filenames.

Edit only the paths in `configs/default.yaml`:

```yaml
data:
  train_root: "D:/datasets/LOLv1/our485"
  test_root: "D:/datasets/LOLv1/eval15"
```

The first 400 sorted pairs from `our485` are used for training. Remaining pairs are assigned to validation, matching the original split.

## Training

```bash
python train.py
```

Alternative configuration:

```bash
python train.py --config configs/default.yaml
```

Default training settings:

| Setting | Value |
|---|---:|
| Image size | 256 × 256 |
| Batch size | 2 |
| Training iterations | 500 |
| Generator learning rate | 0.0002 |
| Discriminator learning rate | 0.0009 |
| Refinement learning rate | 0.0003 |
| Generator loss weight | 0.6 |
| Refinement loss weight | 0.4 |
| Preview interval | 10 iterations |
| Checkpoint interval | 100 iterations |

Outputs:

```text
checkpoints/generator_iter_100.h5
checkpoints/discriminator_iter_100.h5
checkpoints/refinement_model_iter_100.h5
outputs/training_history.csv
outputs/visualizations/
```

## Inference

Single image:

```bash
python infer.py --model checkpoints/generator_iter_500.h5 --input image.png --output outputs/inference/enhanced.png
```

Complete folder:

```bash
python infer.py --model checkpoints/generator_iter_500.h5 --input path/to/input_folder --output outputs/inference
```

## Evaluation

```bash
python evaluate.py --model checkpoints/generator_iter_500.h5 --data-root D:/datasets/LOLv1/eval15 --output-csv outputs/evaluation/metrics.csv
```

The evaluation script reports PSNR, SSIM, MAE, runtime, and parameter count.

## Publication status

The full source code is now publicly released following publication of the paper.

## Citation

Replace this block with the final published BibTeX entry before pushing the repository:

```bibtex
@article{selfrefinegan,
  title   = {Enhancing Low-Light Image Quality with Self-RefineGAN: A Lightweight GAN Approach},
  author  = {Muhammad Atif and co-authors},
  journal = {Journal name},
  year    = {Publication year},
  doi     = {DOI}
}
```

## License

This project is distributed under the MIT License. See `LICENSE`.
