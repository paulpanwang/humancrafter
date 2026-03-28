from __future__ import annotations

from pathlib import Path

from humancrafter.utils import (
    compute_chamfer_distance,
    compute_iou,
    compute_semantic_accuracy,
    summarize_metrics,
)


def compute_metrics(predicted, ground_truth, num_classes=4):
    metrics = {}

    if "point_cloud" in predicted and "point_cloud" in ground_truth:
        metrics["chamfer_distance"] = compute_chamfer_distance(
            predicted["point_cloud"], ground_truth["point_cloud"]
        )

    if "semantic_segmentation" in predicted and "semantic_segmentation" in ground_truth:
        metrics["semantic_accuracy"] = compute_semantic_accuracy(
            predicted["semantic_segmentation"], ground_truth["semantic_segmentation"]
        )
        metrics["iou"] = compute_iou(
            predicted["semantic_segmentation"],
            ground_truth["semantic_segmentation"],
            num_classes=num_classes,
        )

    return metrics


def evaluate_dataset(
    pipeline,
    dataset,
    output_dir=None,
    num_classes=4,
    save_predictions=False,
    save_visualization=True,
):
    per_sample_metrics = []
    output_path = Path(output_dir) if output_dir else None

    for item in dataset:
        sample = item["sample"]
        predicted = pipeline(item["image"])
        metrics = compute_metrics(predicted, item["ground_truth"], num_classes=num_classes)
        per_sample_metrics.append({"name": sample.name, **metrics})

        if save_predictions and output_path is not None:
            pipeline.save_results(
                predicted,
                output_path / sample.name,
                save_visualization=save_visualization,
            )

    summary = summarize_metrics(
        [{key: value for key, value in record.items() if key != "name"} for record in per_sample_metrics]
    )

    if output_path is not None:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyYAML is required to write evaluation summaries"
            ) from exc

        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "metrics.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(summary, handle, sort_keys=False)
        with (output_path / "metrics_per_sample.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(per_sample_metrics, handle, sort_keys=False)

    return summary, per_sample_metrics
