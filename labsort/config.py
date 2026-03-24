"""YAML config loader with sensible defaults."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path.home() / ".labsort"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
DB_PATH = CONFIG_DIR / "index.db"

DEFAULTS: dict[str, Any] = {
    "db_path": str(DB_PATH),
    "hash_size_limit_mb": 500,
    "batch_size": 500,
    "ignore_dirs": [
        ".git", ".svn", ".hg", "__pycache__", ".snakemake",
        "node_modules", ".nextflow", ".DS_Store", ".Trash",
        ".Spotlight-V100", ".fseventsd",
    ],
    "ignore_dir_patterns": [
        "*.app", "*.framework",
    ],
    "ignore_patterns": [
        "*.pyc", ".DS_Store", "Thumbs.db", "~$*", ".~lock.*",
    ],
    "scan_hidden": False,
    "custom_rules": {},
}


def load_config() -> dict[str, Any]:
    """Load config from ~/.labsort/config.yaml, merged over defaults."""
    config = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            user = yaml.safe_load(f) or {}
        # Merge: user values override defaults, lists are replaced not appended
        for key, val in user.items():
            if key == "custom_rules" and isinstance(val, dict):
                config.setdefault("custom_rules", {}).update(val)
            else:
                config[key] = val
    return config


def ensure_config_dir() -> Path:
    """Create ~/.labsort/ if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR
