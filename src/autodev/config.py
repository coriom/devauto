from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import os
import yaml


@dataclass(frozen=True)
class LoadedConfig:
    projects_root: Path
    manager_prompt: str
    dev_prompt: str
    constraints: Dict[str, Any]


def _load_env_file(repo_root: Path, filename: str = ".env") -> None:
    """
    Minimal .env loader (no dependency).
    Lines: KEY=VALUE, ignores comments and blank lines.
    Does not overwrite existing environment variables.
    """
    p = repo_root / filename
    if not p.exists():
        return

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def load_config(repo_root: Path) -> LoadedConfig:
    cfg_path = repo_root / "autodev.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    _load_env_file(repo_root)

    paths = raw.get("paths", {})
    projects_root = repo_root / paths.get("projects_root", "projets")

    prompts_paths = paths.get("prompts", {})
    manager_path = repo_root / prompts_paths.get("manager", "prompts/manager_system.txt")
    dev_path = repo_root / prompts_paths.get("dev", "prompts/dev_system.txt")

    manager_prompt = manager_path.read_text(encoding="utf-8")
    dev_prompt = dev_path.read_text(encoding="utf-8")

    constraints = raw.get("constraints", {})

    return LoadedConfig(
        projects_root=projects_root,
        manager_prompt=manager_prompt,
        dev_prompt=dev_prompt,
        constraints=constraints,
    )
