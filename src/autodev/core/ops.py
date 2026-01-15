from __future__ import annotations

from pathlib import Path
from typing import List

from ..config import LoadedConfig
from ..io import read_text, write_text
from ..models import FileOp
from .policies import c_get, enforce_policies_on_ops, secrets_scan_text


def apply_file_ops(workspace_root: Path, cfg: LoadedConfig, ops: List[FileOp]) -> None:
    enforce_policies_on_ops(ops, cfg)

    max_written_bytes_per_file: int = int(c_get(cfg, "max_written_bytes_per_file", 300_000))
    forbid_keywords: List[str] = c_get(cfg, "forbid_keywords", [])

    # Optional safety: prevent "write" overwriting existing files unless explicitly allowed
    # Set in autodev.yaml -> constraints:
    #   forbid_write_overwrite: true
    forbid_write_overwrite: bool = bool(c_get(cfg, "forbid_write_overwrite", False))

    for op in ops:
        rel = op.path.replace("\\", "/")
        abs_path = workspace_root / rel

        if op.op == "write":
            if op.content is None:
                raise ValueError(f"write op missing content for {rel}")

            if forbid_write_overwrite and abs_path.exists():
                raise ValueError(
                    f"write would overwrite existing file (forbid_write_overwrite=true): {rel}. "
                    f"Use replace for edits."
                )

            if len(op.content.encode("utf-8")) > max_written_bytes_per_file:
                raise ValueError(f"File too large for MVP limit: {rel}")

            scan = secrets_scan_text(op.content, forbid_keywords)
            if scan:
                raise ValueError(f"Secrets scan hit in {rel}: {scan}")

            write_text(workspace_root, rel, op.content)

        elif op.op == "append":
            if op.content is None:
                raise ValueError(f"append op missing content for {rel}")

            try:
                existing = read_text(workspace_root, rel, max_bytes=max_written_bytes_per_file)
            except FileNotFoundError:
                existing = ""

            new = existing + op.content

            if len(new.encode("utf-8")) > max_written_bytes_per_file:
                raise ValueError(
                    f"append would exceed max_written_bytes_per_file for {rel} "
                    f"({max_written_bytes_per_file} bytes)."
                )

            scan = secrets_scan_text(new, forbid_keywords)
            if scan:
                raise ValueError(f"Secrets scan hit in {rel}: {scan}")

            write_text(workspace_root, rel, new)

        elif op.op == "replace":
            if op.find is None or op.replace is None:
                raise ValueError(f"replace op missing find/replace for {rel}")

            txt = read_text(workspace_root, rel, max_bytes=max_written_bytes_per_file)
            if op.find not in txt:
                raise ValueError(f"replace 'find' not found in {rel}")

            new = txt.replace(op.find, op.replace)

            scan = secrets_scan_text(new, forbid_keywords)
            if scan:
                raise ValueError(f"Secrets scan hit in {rel}: {scan}")

            write_text(workspace_root, rel, new)

        else:
            raise ValueError(f"Unknown op: {op.op}")
