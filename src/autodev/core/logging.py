from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..io import ensure_dir
from .constants import EVENTS_FILENAME, RUNS_DIRNAME


def events_path(repo_root: Path) -> Path:
    runs_root = repo_root / RUNS_DIRNAME
    ensure_dir(runs_root)
    return runs_root / EVENTS_FILENAME


def log_event(repo_root: Path, event: Dict[str, Any]) -> None:
    """
    Append one JSON event per line. Minimal, scalable, easy to parse.
    """
    p = events_path(repo_root)
    line = json.dumps(event, ensure_ascii=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
