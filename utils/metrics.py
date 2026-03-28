import numpy as np


def _flatten_points(points):
    points = np.asarray(points)
    if points.size == 0:
        raise ValueError("point cloud must not be empty")
    if points.ndim == 1:
        raise ValueError("point cloud must include a coordinate axis")
    return points.reshape(-1, points.shape[-1])


def compute_chamfer_distance(predicted, ground_truth):
    """
    Compute the symmetric Chamfer distance between two point clouds.

    Args:
        predicted: Predicted point cloud.
        ground_truth: Ground-truth point cloud.

    Returns:
        float: Symmetric Chamfer distance.
    """
    predicted = _flatten_points(predicted)
    ground_truth = _flatten_points(ground_truth)

    dist_pred = np.sqrt(
        np.min(np.sum((predicted[:, None] - ground_truth[None, :]) ** 2, axis=2), axis=1)
    ).mean()
    dist_gt = np.sqrt(
        np.min(np.sum((ground_truth[:, None] - predicted[None, :]) ** 2, axis=2), axis=1)
    ).mean()

    return float((dist_pred + dist_gt) / 2.0)


def compute_iou(predicted, ground_truth, num_classes=None):
    """
    Compute mean IoU for discrete semantic labels.

    Args:
        predicted: Predicted semantic labels.
        ground_truth: Ground-truth semantic labels.
        num_classes: Optional number of classes. If omitted, infer it.

    Returns:
        float: Mean IoU across classes present in either array.
    """
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)

    if predicted.shape != ground_truth.shape:
        raise ValueError("predicted and ground_truth must have the same shape")

    if num_classes is None:
        max_label = int(max(predicted.max(initial=0), ground_truth.max(initial=0)))
        num_classes = max_label + 1

    ious = []
    for class_index in range(num_classes):
        intersection = np.sum((predicted == class_index) & (ground_truth == class_index))
        union = np.sum((predicted == class_index) | (ground_truth == class_index))
        if union > 0:
            ious.append(intersection / union)

    if not ious:
        return 0.0

    return float(np.mean(ious))
