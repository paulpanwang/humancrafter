import numpy as np
import torch
from PIL import Image


def preprocess_image(image, image_size=224):
    """
    Preprocess an input image for HumanCrafter.

    Args:
        image: PIL image instance or path to an image file.
        image_size: Target square image size.

    Returns:
        torch.Tensor: Normalized CHW image tensor.
    """
    if isinstance(image, str):
        image = Image.open(image)

    image = image.convert("RGB")
    image = image.resize((image_size, image_size))

    image = np.asarray(image, dtype=np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )

    return torch.from_numpy(image).permute(2, 0, 1)


def postprocess_results(output):
    """
    Convert model outputs to numpy arrays for downstream use.

    Args:
        output: Model output dictionary.

    Returns:
        dict: Postprocessed inference results.
    """
    point_maps = output["point_maps"].detach().cpu().numpy()
    semantic_labels = output["semantic_labels"].detach().cpu().numpy()
    appearance_features = output["appearance_features"].detach().cpu().numpy()

    return {
        "point_cloud": point_maps,
        "semantic_segmentation": np.argmax(semantic_labels, axis=-1),
        "appearance_features": appearance_features,
    }
