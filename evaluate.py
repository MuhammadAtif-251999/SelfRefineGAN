"""Evaluate a trained generator on paired low-light images."""

from __future__ import annotations

import argparse

import numpy as np

from selfrefine_gan.evaluation.evaluator import evaluate_dataset
from selfrefine_gan.runtime import configure_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Generator .h5 path.")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Folder containing input/ and gt/ subfolders.",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--output-csv",
        default="outputs/evaluation/metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime()

    rows, parameter_count = evaluate_dataset(
        model_path=args.model,
        data_root=args.data_root,
        image_height=args.height,
        image_width=args.width,
        output_csv=args.output_csv,
    )

    for row in rows:
        print(
            f"{row['image']}: PSNR={float(row['psnr']):.4f}, "
            f"SSIM={float(row['ssim']):.4f}, "
            f"MAE={float(row['mae']):.6f}, "
            f"time={float(row['runtime_seconds']):.4f}s"
        )

    print("\nAverage metrics")
    for key in ("psnr", "ssim", "mae", "runtime_seconds"):
        average = np.mean([float(row[key]) for row in rows])
        print(f"{key}: {average:.6f}")
    print(f"Model parameters: {parameter_count:,}")
    print(f"Saved metrics: {args.output_csv}")


if __name__ == "__main__":
    main()
