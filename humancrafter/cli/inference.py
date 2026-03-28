from __future__ import annotations

import argparse
from pathlib import Path

from humancrafter.config import getattr_or, load_config
from humancrafter.datasets import discover_images
from humancrafter.models import HumanCrafterModel, load_checkpoint
from humancrafter.pipeline import HumanCrafterPipeline


def build_parser():
    parser = argparse.ArgumentParser(description="Run HumanCrafter inference on a directory of images.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)

    inference_config = getattr(config, "inference", None)
    device = args.device or getattr_or(inference_config, "device", "cpu")
    image_size = getattr_or(inference_config, "image_size", getattr_or(config.model, "image_size", 256))
    save_visualization = getattr_or(inference_config, "save_visualization", True)
    file_patterns = getattr_or(inference_config, "file_patterns", None)

    model = HumanCrafterModel(config.model)
    checkpoint_path = getattr_or(config.model, "checkpoint_path", None)
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, map_location=device)

    pipeline = HumanCrafterPipeline(model, image_size=image_size, device=device)

    image_paths = discover_images(args.input_dir, patterns=file_patterns)
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {args.input_dir}")

    output_root = Path(args.output_dir)
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        results = pipeline(image_path)
        pipeline.save_results(
            results,
            output_root / image_path.stem,
            save_visualization=save_visualization,
        )

    print(f"Inference completed. Results saved to {output_root}")


if __name__ == "__main__":
    main()
