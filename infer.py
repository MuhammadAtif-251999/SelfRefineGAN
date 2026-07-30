"""Enhance one low-light image or a complete folder."""

from __future__ import annotations

import argparse

from selfrefine_gan.inference.predictor import Predictor
from selfrefine_gan.runtime import configure_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Generator .h5 path.")
    parser.add_argument("--input", required=True, help="Image or folder path.")
    parser.add_argument(
        "--output",
        default="outputs/inference",
        help="Output image or folder path.",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--keep-original-size",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime()
    predictor = Predictor(args.model, args.height, args.width)
    saved_paths = predictor.run(
        args.input,
        args.output,
        keep_original_size=args.keep_original_size,
    )
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
