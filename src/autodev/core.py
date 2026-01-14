from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from .config import load_config, LoadedConfig
from .io import ensure_dir, list_tree, read_text, write_text
from .llm import generate_json
from .models import DevResponse, FileOp, Ticket
from .progress import Progress

RUNS_DIRNAME = "runs"
EVENTS_FILENAME = "events.ndjson"
STATES_DIRNAME = "states"

# Mode B defaults
MAX_MANAGER_RETRIES = 2
MAX_DEV_RETRIES = 2
TOUCH_HISTORY_LIMIT = 50  # stored in state; keep it bounded


# ---------------------------------------------------------------------
# Small JSON helpers
# ---------------------------------------------------------------------
def write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# Run logging (single append-only file)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Run folders (kept minimal for compatibility)
# ---------------------------------------------------------------------
def next_run_dir(repo_root: Path) -> Path:
    runs_root = repo_root / RUNS_DIRNAME
    ensure_dir(runs_root)

    existing = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")])
    n = 1
    if existing:
        last = existing[-1].name.replace("run_", "")
        if last.isdigit():
            n = int(last) + 1
    return runs_root / f"run_{n:04d}"


# ---------------------------------------------------------------------
# Constraints helpers
# ---------------------------------------------------------------------
def _c_get(cfg: LoadedConfig, key: str, default):
    return cfg.constraints.get(key, default)


def _is_allowed(path: str, allowed: List[str]) -> bool:
    return any(path.startswith(prefix) for prefix in allowed) if allowed else True


def _is_blocked(path: str, blocked: List[str]) -> bool:
    return any(path.startswith(prefix) or path == prefix for prefix in blocked)


def secrets_scan_text(text: str, forbid: List[str]) -> List[str]:
    found = []
    upper = text.upper()
    for kw in forbid:
        if kw.upper() in upper:
            found.append(kw)
    return found


def enforce_policies_on_ops(ops: List[FileOp], cfg: LoadedConfig) -> None:
    allowed_paths: List[str] = _c_get(cfg, "allowed_paths", [])
    blocked_paths: List[str] = _c_get(cfg, "blocked_paths", [])

    for op in ops:
        rel = op.path.replace("\\", "/")
        if _is_blocked(rel, blocked_paths):
            raise ValueError(f"Blocked path touched: {rel}")
        if not _is_allowed(rel, allowed_paths):
            raise ValueError(f"Path not in allowed_paths: {rel}")


# ---------------------------------------------------------------------
# remaining_work sanitization (avoid internal/meta tasks)
# ---------------------------------------------------------------------
_STATE_TODO_PATTERNS = [
    r"\bstates?\b",
    r"states[/\\]",
    r"\bstate\.json\b",
    r"\bSTATE\.json\b",
    r"\bupdate\b.*\bstate\b",
    r"\bmettre\s+à\s+jour\b.*\bstate\b",
    r"\bmaj\b.*\bstate\b",
    r"\bprogress\b.*\bstate\b",
]
_STATE_TODO_RE = re.compile("|".join(_STATE_TODO_PATTERNS), flags=re.IGNORECASE)


def sanitize_remaining_work(items: Any) -> List[str]:
    """
    Remove meta/internal tasks (e.g., "update state") and normalize.
    Keeps order, removes duplicates, strips whitespace.
    """
    if not isinstance(items, list):
        return []

    out: List[str] = []
    seen = set()

    for x in items:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s:
            continue

        if _STATE_TODO_RE.search(s):
            continue

        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)

    return out


# ---------------------------------------------------------------------
# Objective path extraction (context boost + mode B validation)
# ---------------------------------------------------------------------
_PATH_LINE_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.-]+[/\\])+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9]+)?)"
)


def _norm_rel_path(p: str) -> Optional[str]:
    if not isinstance(p, str):
        return None
    s = p.replace("\\", "/").strip()
    if not s:
        return None
    if ":" in s:
        return None
    if s.startswith("/"):
        return None
    if ".." in s:
        return None
    if any(c in s for c in ["\n", "\r", "\t"]):
        return None
    return s


def extract_paths_from_text(text: str) -> List[str]:
    """
    Extract relative paths like 'src/main.py', 'src/questions.txt', 'src/testsite/site.html'.
    Returns normalized forward-slash paths, deduped, order-preserving.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    found: List[str] = []
    seen = set()

    for m in _PATH_LINE_RE.finditer(text):
        raw = m.group("path").strip()
        p = _norm_rel_path(raw)
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(p)

    return found


def extract_paths_from_objective(text: str) -> List[str]:
    return extract_paths_from_text(text)


def extract_paths_from_ticket_raw(raw_ticket: Dict[str, Any]) -> List[str]:
    """
    Best-effort extraction of targeted files from ticket content.
    If manager provides files_to_modify, use it; otherwise extract from
    title/goal/plan_steps/definition_of_done/remaining_work/progress_summary.
    """
    if not isinstance(raw_ticket, dict):
        return []

    # Prefer explicit field (mode B)
    files = raw_ticket.get("files_to_modify")
    if isinstance(files, list):
        out: List[str] = []
        seen = set()
        for x in files:
            if not isinstance(x, str):
                continue
            p = _norm_rel_path(x)
            if not p:
                continue
            k = p.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
        if out:
            return out

    # Fallback: extract from text fields
    texts: List[str] = []
    for k in ["title", "goal", "progress_summary"]:
        v = raw_ticket.get(k)
        if isinstance(v, str):
            texts.append(v)

    for k in ["definition_of_done", "plan_steps", "remaining_work"]:
        v = raw_ticket.get(k)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    texts.append(item)

    blob = "\n".join(texts)
    return extract_paths_from_text(blob)


def extract_paths_from_remaining_work(items: List[str]) -> List[str]:
    if not isinstance(items, list):
        return []
    blob = "\n".join([x for x in items if isinstance(x, str)])
    return extract_paths_from_text(blob)


# ---------------------------------------------------------------------
# STATE store: one state file per objective, in projets/<project>/states/
# + Mode B fields: files{}, touch_history[]
# ---------------------------------------------------------------------
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
    """
    Stable ID:
    - If objective_file provided: slugify its stem (human-friendly)
    - Else: short sha256 hash of objective text
    """
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


def _ensure_state_mode_b_fields(st: Dict[str, Any]) -> Dict[str, Any]:
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
        st = _ensure_state_mode_b_fields(st)
        return st
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


# ---------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------
def build_context(
    workspace_root: Path,
    cfg: LoadedConfig,
    focus: str = "",
    *,
    objective_text: str = "",
) -> Tuple[List[str], List[Dict]]:
    """
    Returns (conventions, relevant_files)
    relevant_files are small excerpts used by Manager in the ticket.
    """
    max_tree_files: int = int(_c_get(cfg, "max_tree_files", 2000))
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
        # Mode B governance hint (soft-lock)
        "Mode B: avoid rewriting the same file repeatedly. If a file was already completed/pinned, do NOT touch it unless you explicitly justify it (allow_retouch_pinned + rationale_by_file).",
        "Mode B: prefer adding 'files_to_modify' to the ticket so DEV stays in-scope.",
    ]

    # Pick files in priority order:
    # 1) paths explicitly mentioned in objective (if they exist)
    # 2) focus hits
    # 3) fallback first files
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
        excerpt = excerpt[:1200]
        relevant_files.append({"path": rel, "excerpt": excerpt})

    return conventions, relevant_files


# ---------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------
def make_manager_prompt(
    cfg: LoadedConfig,
    objective: str,
    conventions: List[str],
    relevant_files: List[Dict],
    state: Dict[str, Any],
) -> str:
    payload = {
        "objective": objective,
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
            # Mode B optional fields (raw JSON accepted even if schema ignores extras):
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


# ---------------------------------------------------------------------
# Mode B: ticket validation & repair prompts (Manager/Dev retries)
# ---------------------------------------------------------------------
def _manager_validation_errors(raw_ticket: Dict[str, Any], st: Dict[str, Any]) -> List[str]:
    errs: List[str] = []

    # Required strict fields
    if "progress_summary" not in raw_ticket or not str(raw_ticket.get("progress_summary", "")).strip():
        errs.append("Missing/empty progress_summary (required).")
    if "remaining_work" not in raw_ticket or not isinstance(raw_ticket.get("remaining_work"), list):
        errs.append("Missing remaining_work list (required).")

    # File target extraction (Mode B)
    targets = extract_paths_from_ticket_raw(raw_ticket)
    if not targets:
        errs.append("No target files detected. Provide files_to_modify or include explicit paths in plan_steps/DoD.")
        return errs

    # Pinned enforcement
    allow = raw_ticket.get("allow_retouch_pinned", [])
    allow_set = set()
    if isinstance(allow, list):
        for x in allow:
            if isinstance(x, str):
                p = _norm_rel_path(x)
                if p:
                    allow_set.add(p.lower())

    rationale = raw_ticket.get("rationale_by_file", {})
    rationale_map: Dict[str, str] = {}
    if isinstance(rationale, dict):
        for k, v in rationale.items():
            if isinstance(k, str) and isinstance(v, str):
                pk = _norm_rel_path(k)
                if pk:
                    rationale_map[pk.lower()] = v.strip()

    files_state = st.get("files", {}) if isinstance(st, dict) else {}
    if not isinstance(files_state, dict):
        files_state = {}

    pinned_violations: List[str] = []
    missing_rationale: List[str] = []

    for pth in targets:
        key = pth.lower()
        info = files_state.get(pth) or files_state.get(key)  # tolerate mixed keys
        if isinstance(info, dict) and info.get("pinned") is True:
            if key not in allow_set:
                pinned_violations.append(pth)
            if key in allow_set:
                r = rationale_map.get(key, "")
                if not r:
                    missing_rationale.append(pth)

    if pinned_violations:
        errs.append("Ticket targets pinned file(s) without allow_retouch_pinned: " + ", ".join(pinned_violations))
    if missing_rationale:
        errs.append("Pinned retouch requires rationale_by_file for: " + ", ".join(missing_rationale))

    return errs


def _repair_prompt(base_prompt: str, errors: List[str], last_output: Dict[str, Any]) -> str:
    payload = {
        "validation_errors": errors,
        "last_output": last_output,
        "instruction": "Fix the JSON to satisfy the schema + validation errors. Return ONLY corrected JSON.",
    }
    return base_prompt + "\n\nVALIDATION:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _validate_dev_ops_nonempty(dev_obj: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    ops = dev_obj.get("file_ops")
    if not isinstance(ops, list) or len(ops) == 0:
        errs.append("DevResponse.file_ops must be a non-empty list.")
    return errs


# ---------------------------------------------------------------------
# Apply ops
# ---------------------------------------------------------------------
def apply_file_ops(workspace_root: Path, cfg: LoadedConfig, ops: List[FileOp]) -> None:
    enforce_policies_on_ops(ops, cfg)

    max_written_bytes_per_file: int = int(_c_get(cfg, "max_written_bytes_per_file", 300_000))
    forbid_keywords: List[str] = _c_get(cfg, "forbid_keywords", [])

    for op in ops:
        rel = op.path.replace("\\", "/")

        if op.op == "write":
            if op.content is None:
                raise ValueError(f"write op missing content for {rel}")
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


# ---------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------
def get_workspace_root(repo_root: Path, cfg: LoadedConfig, project: str) -> Path:
    if not project:
        raise ValueError("Project name is required (e.g. --project DeOpt)")
    return cfg.projects_root / project


# ---------------------------------------------------------------------
# Mode B: update per-file state (hashes, pins, history)
# ---------------------------------------------------------------------
def _read_existing_text_safe(workspace_root: Path, rel: str, *, max_bytes: int = 200_000) -> str:
    try:
        return read_text(workspace_root, rel, max_bytes=max_bytes)
    except Exception:
        return ""


def _update_state_files_after_ops(
    *,
    workspace_root: Path,
    st_after: Dict[str, Any],
    run_name: str,
    touched_paths: List[str],
    pin_files: List[str],
    remaining_work: List[str],
) -> None:
    st_after = _ensure_state_mode_b_fields(st_after)
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

    # Apply explicit pins
    for pth in pin_files:
        pn = _norm_rel_path(pth)
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

    # Auto-pin heuristic:
    # If a file was touched AND it's NOT referenced in remaining_work paths, mark pinned.
    for rel in touched_paths:
        rn = rel.replace("\\", "/")
        if rn.lower() in remaining_paths:
            continue
        info = files.get(rn)
        if isinstance(info, dict):
            info["pinned"] = True
            files[rn] = info

    # Bound history
    if len(history) > TOUCH_HISTORY_LIMIT:
        st_after["touch_history"] = history[-TOUCH_HISTORY_LIMIT:]


# ---------------------------------------------------------------------
# Human-in-the-loop run (kept)
# ---------------------------------------------------------------------
def create_run(
    repo_root: Path,
    project: str,
    objective: str,
    focus: str = "",
    objective_file: str = "",
    *,
    mode: str = "human",  # "human" or "auto"
) -> Path:
    """
    mode="human": write prompts + templates
    mode="auto": minimal run dir (meta + ticket/dev/summary only)
    """
    cfg = load_config(repo_root)
    workspace_root = get_workspace_root(repo_root, cfg, project)
    ensure_dir(workspace_root)
    ensure_dir(states_dir(workspace_root))

    run_dir = next_run_dir(repo_root)
    ensure_dir(run_dir)

    st = load_or_init_state(workspace_root, objective=objective, objective_file=objective_file)

    conventions, relevant_files = build_context(workspace_root, cfg, focus=focus, objective_text=objective)
    manager_prompt = make_manager_prompt(cfg, objective, conventions, relevant_files, state=st)

    meta = {
        "ts": utc_now_iso(),
        "project": project,
        "workspace_root": workspace_root.as_posix(),
        "focus": focus,
        "objective_file": objective_file or None,
        "state_id": st.get("state_id"),
    }
    write_json(run_dir / "run_meta.json", meta)

    if mode == "human":
        (run_dir / "manager_prompt.txt").write_text(manager_prompt, encoding="utf-8")
        write_json(run_dir / "ticket.json", {"TODO": "Paste Manager JSON Ticket here"})
        write_json(run_dir / "dev_response.json", {"TODO": "Paste Dev JSON response here"})
    else:
        write_json(run_dir / "ticket.json", {})
        write_json(run_dir / "dev_response.json", {})

    log_event(repo_root, {"ts": meta["ts"], "type": "run_created", "run": run_dir.name, **meta})
    return run_dir


def apply_run(repo_root: Path, run_dir: Path, project: Optional[str] = None) -> None:
    cfg = load_config(repo_root)

    meta = read_json(run_dir / "run_meta.json")
    project_name = project or meta.get("project")
    if not project_name:
        raise ValueError("Project not found. Provide --project or ensure run_meta.json contains it.")

    workspace_root = get_workspace_root(repo_root, cfg, project_name)

    objective_text = meta.get("objective", "")
    objective_file = meta.get("objective_file") or ""
    st = load_or_init_state(workspace_root, objective=objective_text, objective_file=objective_file)

    # Load & validate ticket
    raw_ticket = read_json(run_dir / "ticket.json")
    try:
        ticket = Ticket.model_validate(raw_ticket)
    except ValidationError as e:
        log_event(
            repo_root,
            {"ts": utc_now_iso(), "type": "apply_error", "run": run_dir.name, "error": f"Invalid ticket.json: {e}"},
        )
        raise ValueError(f"Invalid ticket.json: {e}") from e

    # STRICT required fields
    if (
        not isinstance(raw_ticket, dict)
        or "progress_summary" not in raw_ticket
        or not str(raw_ticket.get("progress_summary", "")).strip()
    ):
        raise ValueError("Manager ticket must include a non-empty progress_summary.")
    if (
        not isinstance(raw_ticket, dict)
        or "remaining_work" not in raw_ticket
        or not isinstance(raw_ticket.get("remaining_work"), list)
    ):
        raise ValueError("Manager ticket must include remaining_work as a list (may be empty when completed).")

    # Load & validate dev response
    raw_dev = read_json(run_dir / "dev_response.json")
    try:
        dev = DevResponse.model_validate(raw_dev)
    except ValidationError as e:
        log_event(
            repo_root,
            {"ts": utc_now_iso(), "type": "apply_error", "run": run_dir.name, "error": f"Invalid dev_response.json: {e}"},
        )
        raise ValueError(f"Invalid dev_response.json: {e}") from e

    if dev.ticket_id != ticket.id:
        raise ValueError("dev_response.ticket_id does not match ticket.id")

    # Apply ops
    apply_file_ops(workspace_root, cfg, dev.file_ops)

    # Update state for this objective
    remaining = sanitize_remaining_work(ticket.remaining_work)
    is_complete = len(remaining) == 0

    st_after = dict(st)
    st_after = _ensure_state_mode_b_fields(st_after)
    st_after["last_run"] = run_dir.name
    st_after["progress_summary"] = ticket.progress_summary
    st_after["todo"] = remaining

    # phase sync
    if is_complete:
        st_after["phase"] = "completed"
    else:
        if st_after.get("phase") == "completed":
            st_after["phase"] = "implementation"

    # Shallow merge manager state_update
    if isinstance(ticket.state_update, dict) and ticket.state_update:
        for k, v in ticket.state_update.items():
            st_after[k] = v

    # Mode B: per-file tracking + pinning
    touched_paths = [op.path.replace("\\", "/") for op in dev.file_ops]
    pin_files = raw_ticket.get("pin_files", [])
    if not isinstance(pin_files, list):
        pin_files = []
    _update_state_files_after_ops(
        workspace_root=workspace_root,
        st_after=st_after,
        run_name=run_dir.name,
        touched_paths=touched_paths,
        pin_files=[p for p in pin_files if isinstance(p, str)],
        remaining_work=remaining,
    )

    save_state(workspace_root, st_after)

    log_event(
        repo_root,
        {
            "ts": utc_now_iso(),
            "type": "apply_ok",
            "run": run_dir.name,
            "project": project_name,
            "state_id": st_after.get("state_id"),
            "ticket_id": ticket.id,
            "ops": [{"op": op.op, "path": op.path} for op in dev.file_ops],
        },
    )

    summary = [
        f"# AutoDev Run {run_dir.name}",
        "",
        f"Project: {project_name}",
        f"Workspace: {workspace_root.as_posix()}",
        f"State: {st_after.get('state_id')}",
        "",
        f"Ticket: {ticket.id} — {ticket.title}",
        "",
        "## Summary",
        dev.summary,
        "",
        "## Next remaining_work",
        *([f"- {x}" for x in st_after.get("todo", [])] or ["- (none)"]),
        "",
        "## File ops",
        *([f"- {op.op} {op.path}" for op in dev.file_ops] or ["- (none)"]),
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")


# ---------------------------------------------------------------------
# Full auto mode (API) - one iteration
# ---------------------------------------------------------------------
def auto_run(repo_root: Path, project: str, objective: str, focus: str = "", objective_file: str = "") -> Path:
    """
    One iteration:
    - create minimal run dir
    - load state file for this objective (states/<state_id>.json)
    - manager -> ticket.json (with Mode B validation + retry)
    - dev -> dev_response.json (basic validation + retry)
    - apply -> update state (pins + hashes)
    - append events to runs/events.ndjson
    """
    p = Progress()

    p.tick("Create run folder (+ load objective state)")
    run_dir = create_run(
        repo_root,
        project,
        objective,
        focus=focus,
        objective_file=objective_file,
        mode="auto",
    )

    cfg = load_config(repo_root)
    meta = read_json(run_dir / "run_meta.json")
    workspace_root = Path(meta["workspace_root"])

    st = load_or_init_state(workspace_root, objective=objective, objective_file=objective_file)

    p.tick("Build manager prompt (in-memory)")
    conventions, relevant_files = build_context(workspace_root, cfg, focus=focus, objective_text=objective)
    base_manager_prompt = make_manager_prompt(cfg, objective, conventions, relevant_files, state=st)

    # --------------------
    # Manager with retries (Mode B)
    # --------------------
    p.tick("Manager → API call")
    manager_prompt = base_manager_prompt
    ticket_obj: Dict[str, Any] = {}

    for attempt in range(0, MAX_MANAGER_RETRIES + 1):
        ticket_obj = generate_json("manager", manager_prompt)

        log_event(
            repo_root,
            {
                "ts": utc_now_iso(),
                "type": "manager_ticket",
                "run": run_dir.name,
                "attempt": attempt,
                "ticket": ticket_obj,
            },
        )

        errs = _manager_validation_errors(ticket_obj, st)
        if not errs:
            break

        log_event(
            repo_root,
            {
                "ts": utc_now_iso(),
                "type": "manager_ticket_invalid",
                "run": run_dir.name,
                "attempt": attempt,
                "errors": errs,
            },
        )

        if attempt >= MAX_MANAGER_RETRIES:
            raise ValueError("Manager ticket invalid: " + " | ".join(errs))

        manager_prompt = _repair_prompt(base_manager_prompt, errs, ticket_obj)

    # sanitize remaining_work before persisting & dev prompt
    ticket_obj["remaining_work"] = sanitize_remaining_work(ticket_obj.get("remaining_work", []))

    p.tick("Write ticket.json")
    write_json(run_dir / "ticket.json", ticket_obj)

    # Validate into Ticket model (extras ignored, ok)
    ticket = Ticket.model_validate(ticket_obj)

    # --------------------
    # Dev with retries (basic)
    # --------------------
    p.tick("Build dev prompt (in-memory)")
    base_dev_prompt = make_dev_prompt(cfg, ticket)
    dev_prompt = base_dev_prompt

    p.tick("Dev → API call")
    dev_obj: Dict[str, Any] = {}
    for attempt in range(0, MAX_DEV_RETRIES + 1):
        dev_obj = generate_json("dev", dev_prompt)
        log_event(
            repo_root,
            {
                "ts": utc_now_iso(),
                "type": "dev_response",
                "run": run_dir.name,
                "attempt": attempt,
                "dev_response": dev_obj,
            },
        )

        errs = _validate_dev_ops_nonempty(dev_obj)
        if not errs:
            try:
                _ = DevResponse.model_validate(dev_obj)
                break
            except ValidationError as e:
                errs.append(f"Invalid DevResponse schema: {e}")

        log_event(
            repo_root,
            {
                "ts": utc_now_iso(),
                "type": "dev_response_invalid",
                "run": run_dir.name,
                "attempt": attempt,
                "errors": errs,
            },
        )

        if attempt >= MAX_DEV_RETRIES:
            raise ValueError("Dev response invalid: " + " | ".join(errs))

        dev_prompt = _repair_prompt(base_dev_prompt, errs, dev_obj)

    p.tick("Write dev_response.json")
    write_json(run_dir / "dev_response.json", dev_obj)

    # --------------------
    # Apply
    # --------------------
    p.tick("Apply file ops + update objective state")
    try:
        meta["objective"] = objective
        meta["objective_file"] = objective_file or None
        write_json(run_dir / "run_meta.json", meta)

        apply_run(repo_root, run_dir, project=project)
    except Exception as e:
        log_event(repo_root, {"ts": utc_now_iso(), "type": "apply_error", "run": run_dir.name, "error": str(e)})
        raise

    p.tick("Done")
    return run_dir


# ---------------------------------------------------------------------
# Loop mode (multi-iterations)
# ---------------------------------------------------------------------
def loop_run(
    repo_root: Path,
    project: str,
    objective: str,
    *,
    objective_file: str = "",
    focus: str = "",
    max_iter: int = 10,
    stop_on_empty_todo: bool = True,
    max_consecutive_errors: int = 2,
) -> Dict[str, Any]:
    """
    Repeatedly calls auto_run() up to max_iter times.
    Stops early if the objective state.todo becomes empty (default),
    or if too many consecutive errors happen.

    Returns a small dict summary (for CLI printing / future automation).
    """
    cfg = load_config(repo_root)
    workspace_root = get_workspace_root(repo_root, cfg, project)

    sid = state_id_from_objective(objective, objective_file)

    log_event(
        repo_root,
        {
            "ts": utc_now_iso(),
            "type": "loop_start",
            "project": project,
            "state_id": sid,
            "max_iter": max_iter,
            "stop_on_empty_todo": stop_on_empty_todo,
        },
    )

    consecutive_errors = 0
    runs: List[str] = []

    for i in range(1, max_iter + 1):
        st = load_or_init_state(workspace_root, objective=objective, objective_file=objective_file)
        todo = st.get("todo", [])

        # ✅ stop when todo is empty (and we have at least one successful run)
        if stop_on_empty_todo and isinstance(todo, list) and len(todo) == 0 and st.get("last_run") is not None:
            log_event(
                repo_root,
                {"ts": utc_now_iso(), "type": "loop_stop_done", "project": project, "state_id": sid, "iter": i},
            )
            break

        log_event(
            repo_root,
            {
                "ts": utc_now_iso(),
                "type": "loop_iter_start",
                "project": project,
                "state_id": sid,
                "iter": i,
                "todo_len": len(todo) if isinstance(todo, list) else None,
            },
        )

        try:
            run_dir = auto_run(repo_root, project, objective, focus=focus, objective_file=objective_file)
            runs.append(run_dir.name)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            log_event(
                repo_root,
                {
                    "ts": utc_now_iso(),
                    "type": "loop_iter_error",
                    "project": project,
                    "state_id": sid,
                    "iter": i,
                    "error": str(e),
                    "consecutive_errors": consecutive_errors,
                },
            )
            if consecutive_errors >= max_consecutive_errors:
                log_event(
                    repo_root,
                    {
                        "ts": utc_now_iso(),
                        "type": "loop_stop_errors",
                        "project": project,
                        "state_id": sid,
                        "iter": i,
                        "consecutive_errors": consecutive_errors,
                    },
                )
                break

    st_final = load_or_init_state(workspace_root, objective=objective, objective_file=objective_file)
    todo_final = st_final.get("todo", [])
    summary = {
        "project": project,
        "state_id": sid,
        "iterations_run": len(runs),
        "runs": runs,
        "done": (isinstance(todo_final, list) and len(todo_final) == 0 and st_final.get("last_run") is not None),
        "todo_len": len(todo_final) if isinstance(todo_final, list) else None,
        "last_run": st_final.get("last_run"),
    }

    log_event(repo_root, {"ts": utc_now_iso(), "type": "loop_end", **summary})
    return summary
