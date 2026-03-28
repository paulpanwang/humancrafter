from __future__ import annotations

import numpy as np


def _flatten_points(points):
    points = np.asarray(points)
    if points.size == 0:
        raise ValueError("point cloud must not be empty")
    if points.ndim == 1:
        raise ValueError("point cloud must include a coordinate axis")
    return points.reshape(-1, points.shape[-1])


def compute_chamfer_distance(predicted, ground_truth):
    predicted = _flatten_points(predicted)
    ground_truth = _flatten_points(ground_truth)

    distance_pred = np.sqrt(
        np.min(np.sum((predicted[:, None] - ground_truth[None, :]) ** 2, axis=2), axis=1)
    ).mean()
    distance_gt = np.sqrt(
        np.min(np.sum((ground_truth[:, None] - predicted[None, :]) ** 2, axis=2), axis=1)
    ).mean()

    return float((distance_pred + distance_gt) / 2.0)


def compute_semantic_accuracy(predicted, ground_truth):
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    if predicted.shape != ground_truth.shape:
        raise ValueError("predicted and ground_truth must have the same shape")
    return float(np.mean(predicted == ground_truth))


def compute_iou(predicted, ground_truth, num_classes=None):
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)

    if predicted.shape != ground_truth.shape:
        raise ValueError("predicted and ground_truth must have the same shape")

    if num_classes is None:
        max_label = 0
        if predicted.size:
            max_label = max(max_label, int(predicted.max()))
        if ground_truth.size:
            max_label = max(max_label, int(ground_truth.max()))
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


def summarize_metrics(metrics_list):
    if not metrics_list:
        return {}

    summary = {}
    keys = set()
    for metrics in metrics_list:
        keys.update(metrics.keys())

    for key in sorted(keys):
        values = [metrics[key] for metrics in metrics_list if key in metrics]
        if values:
            summary[key] = float(np.mean(values))

    return summary
