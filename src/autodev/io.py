from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class FileHit:
    path: str
    excerpt: str

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def list_tree(repo_root: Path, max_files: int = 2000) -> List[str]:
    out: List[str] = []
    for p in repo_root.rglob("*"):
        if p.is_dir():
            continue
        # ignore common noise folders
        if any(part in {".git", "runs", "__pycache__", ".venv"} for part in p.parts):
            continue
        out.append(str(p.relative_to(repo_root)).replace("\\", "/"))
        if len(out) >= max_files:
            break
    return sorted(out)

def read_text(repo_root: Path, rel_path: str, max_bytes: int = 200_000) -> str:
    p = repo_root / rel_path
    data = p.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")

def write_text(repo_root: Path, rel_path: str, content: str) -> None:
    p = repo_root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
