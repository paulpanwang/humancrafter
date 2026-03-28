from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


_SEGMENTATION_PALETTE = np.array(
    [
        [0, 0, 0],
        [231, 76, 60],
        [46, 204, 113],
        [52, 152, 219],
        [241, 196, 15],
        [155, 89, 182],
    ],
    dtype=np.uint8,
)


def visualize_segmentation(segmentation, output_path=None):
    segmentation = np.asarray(segmentation)
    if segmentation.ndim > 2:
        segmentation = np.squeeze(segmentation)
    if segmentation.ndim != 2:
        raise ValueError("semantic segmentation must be a 2D label map")

    preview = Image.fromarray(_SEGMENTATION_PALETTE[segmentation % len(_SEGMENTATION_PALETTE)])
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preview.save(output_path)
    return preview


def visualize_results(results, output_path=None):
    if "semantic_segmentation" not in results:
        raise ValueError("results must contain 'semantic_segmentation'")
    return visualize_segmentation(results["semantic_segmentation"], output_path=output_path)
