from humancrafter.utils.metrics import (
    compute_chamfer_distance,
    compute_iou,
    compute_semantic_accuracy,
    summarize_metrics,
)
from humancrafter.utils.visualization import visualize_results, visualize_segmentation

try:
    from humancrafter.utils.data import load_array, postprocess_results, preprocess_image
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    def _missing_torch(*args, **kwargs):
        raise ModuleNotFoundError(
            "torch is required to use preprocess_image and postprocess_results"
        ) from exc

    preprocess_image = _missing_torch
    postprocess_results = _missing_torch
    load_array = _missing_torch


__all__ = [
    "compute_chamfer_distance",
    "compute_iou",
    "compute_semantic_accuracy",
    "load_array",
    "postprocess_results",
    "preprocess_image",
    "summarize_metrics",
    "visualize_results",
    "visualize_segmentation",
]
