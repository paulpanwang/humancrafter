from __future__ import annotations

import argparse

from humancrafter.config import getattr_or, load_config
from humancrafter.datasets import HumanCrafterDataset
from humancrafter.evaluation import evaluate_dataset
from humancrafter.models import HumanCrafterModel, load_checkpoint
from humancrafter.pipeline import HumanCrafterPipeline


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate HumanCrafter predictions against a manifest.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_predictions", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)

    evaluation_config = getattr(config, "evaluation", None)
    device = args.device or getattr_or(evaluation_config, "device", "cpu")
    image_size = getattr_or(evaluation_config, "image_size", getattr_or(config.model, "image_size", 256))
    manifest_path = args.manifest or getattr_or(evaluation_config, "manifest_path", None)
    if manifest_path is None:
        raise ValueError("A dataset manifest is required via --manifest or evaluation.manifest_path")

    model = HumanCrafterModel(config.model)
    checkpoint_path = getattr_or(config.model, "checkpoint_path", None)
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, map_location=device)

    dataset = HumanCrafterDataset(manifest_path)
    pipeline = HumanCrafterPipeline(model, image_size=image_size, device=device)
    summary, _ = evaluate_dataset(
        pipeline,
        dataset,
        output_dir=args.output_dir,
        num_classes=getattr_or(evaluation_config, "num_semantic_classes", 4),
        save_predictions=args.save_predictions
        or getattr_or(evaluation_config, "save_predictions", False),
        save_visualization=getattr_or(evaluation_config, "save_visualization", True),
    )

    if not summary:
        print("Evaluation completed, but no metrics could be computed from the manifest.")
        return

    print("Evaluation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
