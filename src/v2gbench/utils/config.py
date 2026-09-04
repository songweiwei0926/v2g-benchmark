"""Configuration loading and hashing utilities."""

import hashlib
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: str | Path = "config") -> Dict[str, Dict[str, Any]]:
    """Load all YAML config files from a directory."""
    config_dir = Path(config_dir)
    configs = {}
    for yml in sorted(config_dir.glob("*.yaml")):
        configs[yml.stem] = load_config(yml)
    for yml in sorted(config_dir.glob("*.yml")):
        configs[yml.stem] = load_config(yml)
    return configs


def compute_config_hash(config_dir: str | Path = "config") -> str:
    """Compute SHA256 hash of all config files for integrity checking."""
    config_dir = Path(config_dir)
    hasher = hashlib.sha256()
    for yml in sorted(config_dir.glob("*.yaml")):
        hasher.update(yml.read_bytes())
    for yml in sorted(config_dir.glob("*.yml")):
        hasher.update(yml.read_bytes())
    # Also hash subdirectory configs
    for sub in sorted(config_dir.iterdir()):
        if sub.is_dir():
            for yml in sorted(sub.glob("*.yaml")):
                hasher.update(yml.read_bytes())
            for yml in sorted(sub.glob("*.yml")):
                hasher.update(yml.read_bytes())
    return hasher.hexdigest()


def get_project_root() -> Path:
    """Find the project root by looking for run_once.sh."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "run_once.sh").exists():
            return current
        current = current.parent
    return Path.cwd()
