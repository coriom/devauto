from __future__ import annotations

import json
from typing import Any, Dict, List

from .utils import extract_paths_from_ticket_raw, norm_rel_path


def manager_validation_errors(raw_ticket: Dict[str, Any], st: Dict[str, Any]) -> List[str]:
    errs: List[str] = []

    if "progress_summary" not in raw_ticket or not str(raw_ticket.get("progress_summary", "")).strip():
        errs.append("Missing/empty progress_summary (required).")
    if "remaining_work" not in raw_ticket or not isinstance(raw_ticket.get("remaining_work"), list):
        errs.append("Missing remaining_work list (required).")

    targets = extract_paths_from_ticket_raw(raw_ticket)
    if not targets:
        errs.append("No target files detected. Provide files_to_modify or include explicit paths in plan_steps/DoD.")
        return errs

    allow = raw_ticket.get("allow_retouch_pinned", [])
    allow_set = set()
    if isinstance(allow, list):
        for x in allow:
            if isinstance(x, str):
                p = norm_rel_path(x)
                if p:
                    allow_set.add(p.lower())

    rationale = raw_ticket.get("rationale_by_file", {})
    rationale_map: Dict[str, str] = {}
    if isinstance(rationale, dict):
        for k, v in rationale.items():
            if isinstance(k, str) and isinstance(v, str):
                pk = norm_rel_path(k)
                if pk:
                    rationale_map[pk.lower()] = v.strip()

    files_state = st.get("files", {}) if isinstance(st, dict) else {}
    if not isinstance(files_state, dict):
        files_state = {}

    pinned_violations: List[str] = []
    missing_rationale: List[str] = []

    for pth in targets:
        key = pth.lower()
        info = files_state.get(pth) or files_state.get(key)
        if isinstance(info, dict) and info.get("pinned") is True:
            if key not in allow_set:
                pinned_violations.append(pth)
            if key in allow_set:
                if not rationale_map.get(key, ""):
                    missing_rationale.append(pth)

    if pinned_violations:
        errs.append("Ticket targets pinned file(s) without allow_retouch_pinned: " + ", ".join(pinned_violations))
    if missing_rationale:
        errs.append("Pinned retouch requires rationale_by_file for: " + ", ".join(missing_rationale))

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
