from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List

from ..io import ensure_dir, read_text
from .constants import STATES_DIRNAME, TOUCH_HISTORY_LIMIT
from .utils import extract_paths_from_remaining_work, sha256_text, utc_now_iso, write_json, read_json, norm_rel_path


def states_dir(workspace_root: Path) -> Path:
    d = workspace_root / STATES_DIRNAME
    ensure_dir(d)
    return d


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return (s[:80] or "objective")


def state_id_from_objective(objective_text: str, objective_file: str = "") -> str:
    if objective_file:
        stem = Path(objective_file).stem
        return _slug(stem)
    h = hashlib.sha256(objective_text.encode("utf-8")).hexdigest()[:16]
    return f"obj-{h}"


def state_path(workspace_root: Path, state_id: str) -> Path:
    return states_dir(workspace_root) / f"{state_id}.json"


def default_state(objective: str, objective_file: str, state_id: str) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "state_id": state_id,
        "objective": objective,
        "objective_file": objective_file or None,
        "phase": "bootstrapping",
        "done": [],
        "todo": [],
        "decisions": [],
        "last_run": None,
        "progress_summary": "",
        # Mode B
        "files": {},  # path -> {sha256, pinned, touch_count, last_touched_run, last_touched_ts}
        "touch_history": [],  # list of {path, run, ts}
    }


def ensure_state_mode_b_fields(st: Dict[str, Any]) -> Dict[str, Any]:
    if "files" not in st or not isinstance(st.get("files"), dict):
        st["files"] = {}
    if "touch_history" not in st or not isinstance(st.get("touch_history"), list):
        st["touch_history"] = []
    return st


def load_or_init_state(workspace_root: Path, objective: str, objective_file: str = "") -> Dict[str, Any]:
    sid = state_id_from_objective(objective, objective_file)
    p = state_path(workspace_root, sid)

    if not p.exists():
        st = default_state(objective, objective_file, sid)
        write_json(p, st)
        return st

    try:
        st = read_json(p)
        if "state_id" not in st:
            st["state_id"] = sid
        if "objective" not in st:
            st["objective"] = objective
        if "objective_file" not in st:
            st["objective_file"] = objective_file or None
        if "todo" not in st or not isinstance(st.get("todo"), list):
            st["todo"] = []
        return ensure_state_mode_b_fields(st)
    except Exception:
        st = default_state(objective, objective_file, sid)
        write_json(p, st)
        return st


def save_state(workspace_root: Path, st: Dict[str, Any]) -> None:
    sid = st.get("state_id") or state_id_from_objective(
        st.get("objective", ""), st.get("objective_file") or ""
    )
    p = state_path(workspace_root, sid)
    write_json(p, st)


def _read_existing_text_safe(workspace_root: Path, rel: str, *, max_bytes: int = 200_000) -> str:
    try:
        return read_text(workspace_root, rel, max_bytes=max_bytes)
    except Exception:
        return ""


def update_state_files_after_ops(
    *,
    workspace_root: Path,
    st_after: Dict[str, Any],
    run_name: str,
    touched_paths: List[str],
    pin_files: List[str],
    remaining_work: List[str],
) -> None:
    st_after = ensure_state_mode_b_fields(st_after)
    files = st_after["files"]
    history = st_after["touch_history"]

    remaining_paths = set([p.lower() for p in extract_paths_from_remaining_work(remaining_work)])

    for rel in touched_paths:
        rel_norm = rel.replace("\\", "/")
        content = _read_existing_text_safe(workspace_root, rel_norm)
        h = sha256_text(content) if content else ""

        info = files.get(rel_norm)
        if not isinstance(info, dict):
            info = {
                "sha256": "",
                "pinned": False,
                "touch_count": 0,
                "last_touched_run": None,
                "last_touched_ts": None,
            }

        info["sha256"] = h
        info["touch_count"] = int(info.get("touch_count", 0)) + 1
        info["last_touched_run"] = run_name
        info["last_touched_ts"] = utc_now_iso()
        files[rel_norm] = info

        history.append({"path": rel_norm, "run": run_name, "ts": info["last_touched_ts"]})

    # Explicit pins
    for pth in pin_files:
        pn = norm_rel_path(pth)
        if not pn:
            continue
        info = files.get(pn)
        if not isinstance(info, dict):
            info = {
                "sha256": "",
                "pinned": False,
                "touch_count": 0,
                "last_touched_run": None,
                "last_touched_ts": None,
            }
        info["pinned"] = True
        files[pn] = info

    # Auto-pin heuristic: touched but not referenced in remaining_work paths
    for rel in touched_paths:
        rn = rel.replace("\\", "/")
        if rn.lower() in remaining_paths:
            continue
        info = files.get(rn)
        if isinstance(info, dict):
            info["pinned"] = True
            files[rn] = info

    if len(history) > TOUCH_HISTORY_LIMIT:
        st_after["touch_history"] = history[-TOUCH_HISTORY_LIMIT:]
