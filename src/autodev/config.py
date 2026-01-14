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
    Lines: KEY=VALUE (supports optional 'export KEY=VALUE').
    Ignores comments and blank lines.
    Does not overwrite existing environment variables.
    """
    p = repo_root / filename
    if not p.exists():
        return

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # allow: export KEY=VALUE
        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()

        # strip optional quotes
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]

        if k and k not in os.environ:
            os.environ[k] = v


def load_config(repo_root: Path) -> LoadedConfig:
    cfg_path = repo_root / "autodev.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path.as_posix()}")

    raw_text = cfg_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text) or {}
    if not isinstance(raw, dict):
        raise ValueError("Invalid autodev.yaml: expected a YAML mapping/object at root.")

    # Load .env after reading YAML (so YAML can point to prompt paths, but env keys are ready for llm.py)
    _load_env_file(repo_root)

    paths = raw.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}

    projects_root = repo_root / str(paths.get("projects_root", "projets"))

    prompts_paths = paths.get("prompts", {})
    if not isinstance(prompts_paths, dict):
        prompts_paths = {}

    manager_rel = str(prompts_paths.get("manager", "prompts/manager_system.txt"))
    dev_rel = str(prompts_paths.get("dev", "prompts/dev_system.txt"))

    manager_path = repo_root / manager_rel
    dev_path = repo_root / dev_rel

    if not manager_path.is_file():
        raise FileNotFoundError(f"Manager prompt not found: {manager_path.as_posix()}")
    if not dev_path.is_file():
        raise FileNotFoundError(f"Dev prompt not found: {dev_path.as_posix()}")

    manager_prompt = manager_path.read_text(encoding="utf-8")
    dev_prompt = dev_path.read_text(encoding="utf-8")

    constraints = raw.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}

    return LoadedConfig(
        projects_root=projects_root,
        manager_prompt=manager_prompt,
        dev_prompt=dev_prompt,
        constraints=constraints,
    )
