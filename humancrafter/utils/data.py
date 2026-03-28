from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError("torch is required to use humancrafter.utils.data")


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def preprocess_image(image, image_size=256):
    """
    Convert a PIL image or image path to a normalized CHW tensor.
    """
    _require_torch()

    if isinstance(image, (str, Path)):
        image = Image.open(image)

    image = image.convert("RGB")
    image = image.resize((image_size, image_size))

    image = np.asarray(image, dtype=np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )

    return torch.from_numpy(image).permute(2, 0, 1).float()


def postprocess_results(output):
    """
    Convert raw model outputs into numpy arrays for downstream consumers.
    """
    point_maps = _to_numpy(output["point_maps"])
    semantic_logits = _to_numpy(output["semantic_labels"])
    appearance_features = _to_numpy(output["appearance_features"])

    return {
        "point_cloud": point_maps,
        "semantic_segmentation": np.argmax(semantic_logits, axis=-1),
        "appearance_features": appearance_features,
    }


def load_array(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError(f"{path} must contain exactly one array when used as ground truth")
        return data[data.files[0]]
    return np.load(path)
