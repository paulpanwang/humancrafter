from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from humancrafter.utils import postprocess_results, preprocess_image, visualize_results


class HumanCrafterPipeline:
    def __init__(self, model, image_size=256, device="cpu"):
        if torch is None:
            raise ModuleNotFoundError("torch is required to run HumanCrafterPipeline")

        self.model = model.to(device)
        self.image_size = image_size
        self.device = device
        self.model.eval()

    def __call__(self, image):
        image_tensor = preprocess_image(image, image_size=self.image_size).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        return self._squeeze_batch_dimension(postprocess_results(outputs))

    def _squeeze_batch_dimension(self, results):
        squeezed = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == 1:
                squeezed[key] = value[0]
            else:
                squeezed[key] = value
        return squeezed

    def save_results(self, results, output_dir, save_visualization=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        serializable = {}
        for key, value in results.items():
            array_value = np.asarray(value)
            serializable[key] = array_value
            np.save(output_dir / f"{key}.npy", array_value)

        np.savez_compressed(output_dir / "results.npz", **serializable)

        if save_visualization and "semantic_segmentation" in serializable:
            visualize_results(serializable, output_path=output_dir / "semantic_preview.png")

        metadata = {
            "available_keys": sorted(serializable.keys()),
            "shapes": {key: list(value.shape) for key, value in serializable.items()},
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
