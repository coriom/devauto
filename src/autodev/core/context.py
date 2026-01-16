from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import LoadedConfig
from ..io import list_tree, read_text
from ..models import Ticket
from .constants import STATES_DIRNAME
from .policies import c_get
from .utils import extract_paths_from_objective


def build_context(
    workspace_root: Path,
    cfg: LoadedConfig,
    focus: str = "",
    *,
    objective_text: str = "",
) -> Tuple[List[str], List[Dict]]:
    max_tree_files: int = int(c_get(cfg, "max_tree_files", 2000))
    tree = list_tree(workspace_root, max_files=max_tree_files)
    tree_norm = [p.replace("\\", "/") for p in tree]
    tree_set = set(tree_norm)

    conventions = [
        "Return ONLY JSON (no prose) for Ticket and DevResponse.",
        "Use file_ops (write/append/replace).",
        "Keep changes small and localized.",
        f"Workspace root is: {workspace_root.as_posix()}",
        f"STATE files live in: {(workspace_root / STATES_DIRNAME).as_posix()}",
        "The objective is the source of truth; STATE is a progress tracker/cache. If STATE is missing items, reconstruct remaining_work from the objective.",
        "Never add state bookkeeping tasks to remaining_work. Orchestrator handles state persistence.",
        "Mode B: avoid rewriting the same file repeatedly. If a file was already completed/pinned, do NOT touch it unless you explicitly justify it (allow_retouch_pinned + rationale_by_file).",
        "Mode B: prefer adding 'files_to_modify' to the ticket so DEV stays in-scope.",
        # Patch governance
        "If PATCH_TEXT is provided in INPUT, treat it as highest priority constraints/changes for this iteration, but do not violate Mode B rules.",
    ]

    picked: List[str] = []
    relevant_files: List[Dict] = []

    mentioned = extract_paths_from_objective(objective_text)
    for pth in mentioned:
        p_norm = pth.replace("\\", "/")
        if p_norm in tree_set and p_norm not in picked:
            picked.append(p_norm)
        if len(picked) >= 6:
            break

    if len(picked) < 6 and focus:
        for rel in tree_norm:
            if rel in picked:
                continue
            if focus.lower() in rel.lower():
                picked.append(rel)
            if len(picked) >= 6:
                break

    if len(picked) < 6:
        for rel in tree_norm:
            if rel in picked:
                continue
            picked.append(rel)
            if len(picked) >= 6:
                break

    for rel in picked[:6]:
        try:
            excerpt = read_text(workspace_root, rel, max_bytes=60_000)
        except Exception:
            continue
        relevant_files.append({"path": rel, "excerpt": excerpt[:1200]})

    return conventions, relevant_files


def make_manager_prompt(
    cfg: LoadedConfig,
    objective: str,
    conventions: List[str],
    relevant_files: List[Dict],
    state: Dict,
    *,
    patch_text: str = "",
    patch_file: str = "",
) -> str:
    """
    patch_text: optional prioritized modification instructions (already loaded as text)
    patch_file: optional path (string) used only for traceability in the prompt payload
    """
    payload = {
        "objective": objective,
        "patch_text": patch_text or "",
        "patch_file": patch_file or "",
        "state": state,
        "conventions": conventions,
        "relevant_files": relevant_files,
        "ticket_schema_hint": {
            "schema_version": "1.0",
            "id": "T-YYYY-NNNN",
            "title": "Short title",
            "goal": "Outcome",
            "definition_of_done": ["..."],
            "plan_steps": ["..."],
            "labels": ["optional"],
            "progress_summary": "Where we are now (must be non-empty)",
            "remaining_work": [
                "What remains after this ticket (explicit, must be present even if empty at completion). "
                "Do NOT include internal state/update tasks."
            ],
            "state_update": {"optional": "partial state update for this objective state file"},
            # Mode B optional fields (extras accepted even if Ticket ignores extras)
            "files_to_modify": ["Optional but recommended: explicit file targets for this ticket"],
            "rationale_by_file": {"path": "why this file must be touched now"},
            "allow_retouch_pinned": ["If you must touch pinned files, list them here"],
            "pin_files": ["If you consider some files stable after this ticket, list them here"],
        },
    }
    return cfg.manager_prompt.strip() + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def make_dev_prompt(cfg: LoadedConfig, ticket: Ticket) -> str:
    payload = ticket.model_dump()
    payload["dev_schema_hint"] = {
        "schema_version": "1.0",
        "ticket_id": ticket.id,
        "summary": "What you did",
        "file_ops": [
            {"op": "write", "path": "src/Foo.sol", "content": "contract Foo { }"},
            {"op": "replace", "path": "src/Bar.sol", "find": "old", "replace": "new"},
        ],
    }
    return cfg.dev_prompt.strip() + "\n\nTICKET:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
