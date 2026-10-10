"""YAML config loading with single-parent inheritance.

A config may name a parent with ``base: <file>.yaml`` (resolved relative to the config's
directory). The parent is loaded first and the child is deep-merged on top, so a variant
config only lists what it changes. The returned dict is fully resolved and has no ``base`` key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is invalid."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` merged recursively into ``base``.

    Nested dicts are merged key by key; any other value in ``override`` replaces the base value.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ConfigError(f"Cyclic config inheritance: {chain}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping.")
    parent = data.pop("base", None)
    if parent is None:
        return data
    return deep_merge(_load_raw(path.parent / parent, (*seen, path)), data)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolve ``base`` inheritance and validate the seed fields.

    Every config must define ``name`` and either ``seed`` (int) or ``seeds`` (non-empty list
    of ints).
    """
    path = Path(path)
    config = _load_raw(path, ())
    if "name" not in config:
        raise ConfigError(f"{path}: missing required key 'name'.")
    seed, seeds = config.get("seed"), config.get("seeds")
    if seed is None and seeds is None:
        raise ConfigError(f"{path}: must define 'seed' or 'seeds'.")
    if seed is not None and not isinstance(seed, int):
        raise ConfigError(f"{path}: 'seed' must be an int.")
    if seeds is not None and (
        not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) for s in seeds)
    ):
        raise ConfigError(f"{path}: 'seeds' must be a non-empty list of ints.")
    return config
