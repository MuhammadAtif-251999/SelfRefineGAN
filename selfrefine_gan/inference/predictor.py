"""Single-image and folder inference with a saved generator."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from selfrefine_gan.utils.image_io import (
    load_keras_model,
    postprocess_prediction,
    preprocess_pil_image,
)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class Predictor:
    def __init__(self, model_path: str | Path, height: int, width: int) -> None:
        self.model = load_keras_model(model_path)
        self.height = height
        self.width = width

    @staticmethod
    def collect_inputs(path: str | Path) -> list[Path]:
        path = Path(path)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]
        if path.is_dir():
            images = [
                item
                for item in sorted(path.iterdir())
                if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            if images:
                return images
        raise FileNotFoundError(f"No supported input images found at: {path}")

    def enhance(
        self,
        image_path: str | Path,
        keep_original_size: bool = True,
    ) -> Image.Image:
        image_path = Path(image_path)
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            original_size = source.size
            batch = preprocess_pil_image(source, self.width, self.height)

        prediction = self.model.predict(batch, verbose=0)
        return postprocess_prediction(
            prediction,
            original_size if keep_original_size else None,
        )

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        keep_original_size: bool = True,
    ) -> list[Path]:
        input_path = Path(input_path)
        output_path = Path(output_path)
        images = self.collect_inputs(input_path)
        multiple = input_path.is_dir() or len(images) > 1

        if multiple:
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        for image_path in images:
            enhanced = self.enhance(image_path, keep_original_size)
            destination = output_path / image_path.name if multiple else output_path
            enhanced.save(destination)
            saved_paths.append(destination)
        return saved_paths
