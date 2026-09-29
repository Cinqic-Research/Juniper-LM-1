"""Resolve machine-specific locations from configs/local.paths.toml."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_FILE = REPO_ROOT / "configs" / "local.paths.toml"


def load_paths(path: str | os.PathLike | None = None) -> dict:
    """Load the local paths file (override with $JUNIPER_PATHS)."""
    path = Path(path or os.environ.get("JUNIPER_PATHS", DEFAULT_PATHS_FILE))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; copy configs/local.paths.example.toml and edit it."
        )
    with path.open("rb") as f:
        return tomllib.load(f)
