import os

import numpy as np
from PIL import Image


_SEGMENTATION_PALETTE = np.array(
    [
        [0, 0, 0],
        [231, 76, 60],
        [46, 204, 113],
        [52, 152, 219],
        [241, 196, 15],
    ],
    dtype=np.uint8,
)


def visualize_results(results, output_path=None):
    """
    Render a semantic segmentation preview image from model results.

    Args:
        results: Inference result dictionary.
        output_path: Optional destination path for the preview image.

    Returns:
        PIL.Image.Image: Rendered preview image.
    """
    if "semantic_segmentation" not in results:
        raise ValueError("results must contain 'semantic_segmentation'")

    segmentation = np.asarray(results["semantic_segmentation"])
    if segmentation.ndim > 2:
        segmentation = np.squeeze(segmentation)
    if segmentation.ndim != 2:
        raise ValueError("semantic_segmentation must be a 2D label map")

    preview = Image.fromarray(_SEGMENTATION_PALETTE[segmentation % len(_SEGMENTATION_PALETTE)])

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        preview.save(output_path)

    return preview
