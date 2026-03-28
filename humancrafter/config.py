from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _to_dict(value):
    if isinstance(value, SimpleNamespace):
        return {key: _to_dict(item) for key, item in vars(value).items()}
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    return value


def load_config(path):
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to load HumanCrafter config files") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _to_namespace(data)


def dump_config(config, path):
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to write HumanCrafter config files") from exc

    config_path = Path(path)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_dict(config), handle, sort_keys=False)


def getattr_or(namespace, key, default=None):
    if namespace is None:
        return default
    return getattr(namespace, key, default)
