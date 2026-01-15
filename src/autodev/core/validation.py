from __future__ import annotations

import json
from typing import Any, Dict, List, Set, Tuple

from .utils import extract_paths_from_ticket_raw, norm_rel_path


def _norm_list_paths(items: Any) -> Tuple[List[str], List[str]]:
    """
    Returns (normalized_paths, errors)
    """
    errs: List[str] = []
    out: List[str] = []
    seen: Set[str] = set()

    if not isinstance(items, list):
        return [], ["Expected a list of paths."]

    for x in items:
        if not isinstance(x, str):
            continue
        p = norm_rel_path(x)
        if not p:
            errs.append(f"Invalid relative path: {x!r}")
            continue
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)

    return out, errs


def manager_validation_errors(raw_ticket: Dict[str, Any], st: Dict[str, Any]) -> List[str]:
    """
    Mode B hard validation:
    - progress_summary present and non-empty
    - remaining_work list present
    - files_to_modify list present + non-empty
    - rationale_by_file dict covers every file_to_modify
    - allow_retouch_pinned subset of files_to_modify
    - pinned enforcement via state.files[].pinned
    """
    errs: List[str] = []

    # Required strict fields
    if "progress_summary" not in raw_ticket or not str(raw_ticket.get("progress_summary", "")).strip():
        errs.append("Missing/empty progress_summary (required).")
    if "remaining_work" not in raw_ticket or not isinstance(raw_ticket.get("remaining_work"), list):
        errs.append("Missing remaining_work list (required).")

    # Mode B required: files_to_modify must exist and be non-empty
    files_to_modify_raw = raw_ticket.get("files_to_modify")
    files_to_modify, path_errs = _norm_list_paths(files_to_modify_raw)
    errs.extend([f"files_to_modify: {e}" for e in path_errs])

    if not files_to_modify:
        errs.append("files_to_modify must be a non-empty list of relative paths (Mode B required).")
        return errs

    # rationale_by_file must cover each file_to_modify
    rationale = raw_ticket.get("rationale_by_file")
    if not isinstance(rationale, dict):
        errs.append("rationale_by_file must be an object mapping file -> reason (Mode B required).")
        return errs

    missing_rationale_keys: List[str] = []
    empty_rationale_keys: List[str] = []

    # normalize rationale keys
    rationale_map: Dict[str, str] = {}
    for k, v in rationale.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        pk = norm_rel_path(k)
        if not pk:
            continue
        rationale_map[pk.lower()] = v.strip()

    for p in files_to_modify:
        rv = rationale_map.get(p.lower())
        if rv is None:
            missing_rationale_keys.append(p)
        elif not rv:
            empty_rationale_keys.append(p)

    if missing_rationale_keys:
        errs.append(
            "rationale_by_file missing keys for: " + ", ".join(missing_rationale_keys)
        )
    if empty_rationale_keys:
        errs.append(
            "rationale_by_file has empty reasons for: " + ", ".join(empty_rationale_keys)
        )

    # allow_retouch_pinned must be subset of files_to_modify
    allow_raw = raw_ticket.get("allow_retouch_pinned", [])
    allow, allow_errs = _norm_list_paths(allow_raw)
    errs.extend([f"allow_retouch_pinned: {e}" for e in allow_errs])

    files_set = {p.lower() for p in files_to_modify}
    allow_set = {p.lower() for p in allow}

    not_subset = [p for p in allow if p.lower() not in files_set]
    if not_subset:
        errs.append("allow_retouch_pinned must be a subset of files_to_modify. Invalid: " + ", ".join(not_subset))

    # Pinned enforcement based on state
    files_state = st.get("files", {}) if isinstance(st, dict) else {}
    if not isinstance(files_state, dict):
        files_state = {}

    pinned_violations: List[str] = []
    missing_pinned_rationale: List[str] = []

    # If a file is pinned AND in files_to_modify, it must be in allow_retouch_pinned and have rationale (already checked).
    for p in files_to_modify:
        info = files_state.get(p) or files_state.get(p.lower())
        if isinstance(info, dict) and info.get("pinned") is True:
            if p.lower() not in allow_set:
                pinned_violations.append(p)
            else:
                # allowed retouch still requires rationale (already covered), but keep explicit error if missing
                if not rationale_map.get(p.lower(), ""):
                    missing_pinned_rationale.append(p)

    if pinned_violations:
        errs.append(
            "Ticket targets pinned file(s) without allow_retouch_pinned: " + ", ".join(pinned_violations)
        )
    if missing_pinned_rationale:
        errs.append(
            "Pinned retouch requires rationale_by_file for: " + ", ".join(missing_pinned_rationale)
        )

    # Extra: ensure we can detect targets from ticket (for backward compatibility)
    # But in Mode B, the canonical target list is files_to_modify.
    targets = extract_paths_from_ticket_raw(raw_ticket)
    if not targets:
        errs.append("No target files detected (internal). Ensure files_to_modify is present.")
        return errs

    return errs


def repair_prompt(base_prompt: str, errors: List[str], last_output: Dict[str, Any]) -> str:
    payload = {
        "validation_errors": errors,
        "last_output": last_output,
        "instruction": "Fix the JSON to satisfy the schema + validation errors. Return ONLY corrected JSON.",
    }
    return base_prompt + "\n\nVALIDATION:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def validate_dev_ops_nonempty(dev_obj: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    ops = dev_obj.get("file_ops")
    if not isinstance(ops, list) or len(ops) == 0:
        errs.append("DevResponse.file_ops must be a non-empty list.")
    return errs


def validate_dev_ops_scope(dev_obj: Dict[str, Any], raw_ticket: Dict[str, Any], *, st: Dict[str, Any] | None = None) -> List[str]:
    """
    Hard gate: DEV must ONLY touch Ticket.files_to_modify.
    Also enforces pinned retouch: pinned file can be touched only if in allow_retouch_pinned.
    """
    errs: List[str] = []

    files_to_modify, _ = _norm_list_paths(raw_ticket.get("files_to_modify"))
    if not files_to_modify:
        # Manager validation should have caught it; keep defensive.
        errs.append("Ticket missing files_to_modify; cannot validate DEV scope.")
        return errs

    allowed_set = {p.lower() for p in files_to_modify}

    allow_pinned, _ = _norm_list_paths(raw_ticket.get("allow_retouch_pinned", []))
    allow_pinned_set = {p.lower() for p in allow_pinned}

    ops = dev_obj.get("file_ops")
    if not isinstance(ops, list):
        errs.append("DevResponse.file_ops missing/invalid.")
        return errs

    touched: List[str] = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        path = op.get("path")
        if not isinstance(path, str):
            continue
        p = norm_rel_path(path)
        if not p:
            errs.append(f"DevResponse contains invalid path: {path!r}")
            continue
        touched.append(p)

        if p.lower() not in allowed_set:
            errs.append(f"DEV touched file not in Ticket.files_to_modify: {p}")

    # pinned enforcement from state (if provided)
    if st is not None and isinstance(st, dict):
        files_state = st.get("files", {})
        if isinstance(files_state, dict):
            for p in touched:
                info = files_state.get(p) or files_state.get(p.lower())
                if isinstance(info, dict) and info.get("pinned") is True:
                    if p.lower() not in allow_pinned_set:
                        errs.append(f"DEV touched pinned file without allow_retouch_pinned: {p}")

    return errs
