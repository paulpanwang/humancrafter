# Dataset Manifest

HumanCrafter uses a simple manifest file for evaluation data so the repository can stay lightweight.

Each manifest entry should provide at least an `image_path`. Optional supervision arrays can be attached per sample:

- `point_cloud_path`: `.npy` or single-array `.npz`
- `semantic_path`: `.npy` or single-array `.npz`
- `appearance_path`: `.npy` or single-array `.npz`

Example:

```json
[
  {
    "name": "subject_0001",
    "image_path": "../data/subject_0001.png",
    "point_cloud_path": "../data/subject_0001_point_cloud.npy",
    "semantic_path": "../data/subject_0001_semantic.npy"
  }
]
```

Relative paths are resolved from the manifest file location.
