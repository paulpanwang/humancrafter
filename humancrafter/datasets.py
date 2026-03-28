from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

from humancrafter.utils import load_array


@dataclass
class HumanSample:
    name: str
    image_path: Path
    point_cloud_path: Optional[Path] = None
    semantic_path: Optional[Path] = None
    appearance_path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)


def _resolve_path(base_dir: Path, value):
    if value is None:
        return None
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved


def _load_manifest_data(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as handle:
        if manifest_path.suffix.lower() in {".yml", ".yaml"}:
            try:
                import yaml
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "PyYAML is required to load YAML dataset manifests"
                ) from exc
            return yaml.safe_load(handle) or []
        return json.load(handle)


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    entries = _load_manifest_data(manifest_path)

    samples = []
    for index, entry in enumerate(entries):
        name = entry.get("name") or Path(entry["image_path"]).stem or f"sample_{index:04d}"
        samples.append(
            HumanSample(
                name=name,
                image_path=_resolve_path(manifest_dir, entry["image_path"]),
                point_cloud_path=_resolve_path(manifest_dir, entry.get("point_cloud_path")),
                semantic_path=_resolve_path(manifest_dir, entry.get("semantic_path")),
                appearance_path=_resolve_path(manifest_dir, entry.get("appearance_path")),
                metadata=entry.get("metadata", {}),
            )
        )

    return samples


def discover_images(input_dir, patterns: Optional[Iterable[str]] = None):
    input_dir = Path(input_dir)
    patterns = list(patterns or ["*.jpg", "*.jpeg", "*.png", "*.webp"])

    image_paths = []
    for pattern in patterns:
        image_paths.extend(sorted(input_dir.glob(pattern)))

    deduplicated = sorted({path.resolve(): path.resolve() for path in image_paths}.values())
    return deduplicated


class HumanCrafterDataset:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        self.samples = load_manifest(self.manifest_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        ground_truth = {}

        if sample.point_cloud_path:
            ground_truth["point_cloud"] = load_array(sample.point_cloud_path)
        if sample.semantic_path:
            ground_truth["semantic_segmentation"] = load_array(sample.semantic_path)
        if sample.appearance_path:
            ground_truth["appearance_features"] = load_array(sample.appearance_path)

        return {
            "sample": sample,
            "image": image,
            "ground_truth": ground_truth,
        }

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
