from __future__ import annotations

from typing import List

from ..config import LoadedConfig
from ..models import FileOp


def c_get(cfg: LoadedConfig, key: str, default):
    return cfg.constraints.get(key, default)


def is_allowed(path: str, allowed: List[str]) -> bool:
    return any(path.startswith(prefix) for prefix in allowed) if allowed else True


def is_blocked(path: str, blocked: List[str]) -> bool:
    return any(path.startswith(prefix) or path == prefix for prefix in blocked)


def secrets_scan_text(text: str, forbid: List[str]) -> List[str]:
    found = []
    upper = text.upper()
    for kw in forbid:
        if kw.upper() in upper:
            found.append(kw)
    return found


def enforce_policies_on_ops(ops: List[FileOp], cfg: LoadedConfig) -> None:
    allowed_paths: List[str] = c_get(cfg, "allowed_paths", [])
    blocked_paths: List[str] = c_get(cfg, "blocked_paths", [])

    for op in ops:
        rel = op.path.replace("\\", "/")
        if is_blocked(rel, blocked_paths):
            raise ValueError(f"Blocked path touched: {rel}")
        if not is_allowed(rel, allowed_paths):
            raise ValueError(f"Path not in allowed_paths: {rel}")
