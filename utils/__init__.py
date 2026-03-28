from .metrics import compute_chamfer_distance, compute_iou
from .visualization import visualize_results

try:
    from .data import preprocess_image, postprocess_results
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    def _missing_torch(*args, **kwargs):
        raise ModuleNotFoundError(
            "torch is required to use utils.preprocess_image and utils.postprocess_results"
        ) from exc

    preprocess_image = _missing_torch
    postprocess_results = _missing_torch

__all__ = [
    "preprocess_image",
    "postprocess_results",
    "compute_chamfer_distance",
    "compute_iou",
    "visualize_results",
]
