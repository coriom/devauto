from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from pydantic import ValidationError

from .config import load_config, LoadedConfig
from .models import Ticket, DevResponse, FileOp
from .io import list_tree, read_text, write_text, ensure_dir
from .llm import generate_json
from .progress import Progress

RUNS_DIRNAME = "runs"


# ---------- Run store ----------
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


def write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- Constraints helpers ----------
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


# ---------- Context building ----------
def build_context(workspace_root: Path, cfg: LoadedConfig, focus: str = "") -> Tuple[List[str], List[Dict]]:
    """
    Returns (conventions, relevant_files)
    relevant_files are small excerpts used by Manager in the ticket.
    """
    max_tree_files: int = int(_c_get(cfg, "max_tree_files", 2000))
    tree = list_tree(workspace_root, max_files=max_tree_files)

    conventions = [
        "Return ONLY JSON (no prose) for Ticket and DevResponse.",
        "Use file_ops (write/append/replace).",
        "Keep changes small and localized.",
        f"Workspace root is: {workspace_root.as_posix()}",
    ]

    relevant_files: List[Dict] = []
    if focus:
        hits = [p for p in tree if focus.lower() in p.lower()][:6]
    else:
        hits = tree[:6]

    for rel in hits:
        try:
            excerpt = read_text(workspace_root, rel, max_bytes=60_000)
        except Exception:
            continue
        excerpt = excerpt[:1200]
        relevant_files.append({"path": rel, "excerpt": excerpt})

    return conventions, relevant_files


# ---------- Prompt assembly ----------
def make_manager_prompt(cfg: LoadedConfig, objective: str, conventions: List[str], relevant_files: List[Dict]) -> str:
    payload = {
        "objective": objective,
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


# ---------- Apply ops ----------
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


# ---------- Workspace resolution ----------
def get_workspace_root(repo_root: Path, cfg: LoadedConfig, project: str) -> Path:
    if not project:
        raise ValueError("Project name is required (e.g. --project DeOpt)")
    return cfg.projects_root / project


# ---------- Human-in-the-loop run ----------
def create_run(repo_root: Path, project: str, objective: str, focus: str = "") -> Path:
    """
    Creates a run folder with prompts + empty json templates.
    User pastes Manager and Dev JSON outputs into ticket.json and dev_response.json,
    then call apply_run().
    """
    cfg = load_config(repo_root)
    workspace_root = get_workspace_root(repo_root, cfg, project)
    ensure_dir(workspace_root)

    run_dir = next_run_dir(repo_root)
    ensure_dir(run_dir)

    conventions, relevant_files = build_context(workspace_root, cfg, focus=focus)
    manager_prompt = make_manager_prompt(cfg, objective, conventions, relevant_files)

    (run_dir / "manager_prompt.txt").write_text(manager_prompt, encoding="utf-8")
    write_json(run_dir / "ticket.json", {"TODO": "Paste Manager JSON Ticket here"})
    write_json(run_dir / "dev_response.json", {"TODO": "Paste Dev JSON response here"})

    meta = {
        "objective": objective,
        "focus": focus,
        "project": project,
        "workspace_root": workspace_root.as_posix(),
    }
    write_json(run_dir / "run_meta.json", meta)

    return run_dir


def apply_run(repo_root: Path, run_dir: Path, project: Optional[str] = None) -> None:
    cfg = load_config(repo_root)

    meta = read_json(run_dir / "run_meta.json")
    project_name = project or meta.get("project")
    if not project_name:
        raise ValueError("Project not found. Provide --project or ensure run_meta.json contains it.")

    workspace_root = get_workspace_root(repo_root, cfg, project_name)

    # Load & validate ticket
    raw_ticket = read_json(run_dir / "ticket.json")
    try:
        ticket = Ticket.model_validate(raw_ticket)
    except ValidationError as e:
        raise ValueError(f"Invalid ticket.json: {e}") from e

    # Build dev prompt for traceability
    dev_prompt = make_dev_prompt(cfg, ticket)
    (run_dir / "dev_prompt.txt").write_text(dev_prompt, encoding="utf-8")

    # Load & validate dev response
    raw_dev = read_json(run_dir / "dev_response.json")
    try:
        dev = DevResponse.model_validate(raw_dev)
    except ValidationError as e:
        raise ValueError(f"Invalid dev_response.json: {e}") from e

    if dev.ticket_id != ticket.id:
        raise ValueError("dev_response.ticket_id does not match ticket.id")

    # Apply ops into projets/<project>/*
    apply_file_ops(workspace_root, cfg, dev.file_ops)

    # Write summary
    summary = [
        f"# AutoDev Run {run_dir.name}",
        "",
        f"Project: {project_name}",
        f"Workspace: {workspace_root.as_posix()}",
        "",
        f"Ticket: {ticket.id} — {ticket.title}",
        "",
        "## Summary",
        dev.summary,
        "",
        "## Assumptions",
        *([f"- {a}" for a in dev.assumptions] or ["- (none)"]),
        "",
        "## File ops",
        *([f"- {op.op} {op.path}" for op in dev.file_ops] or ["- (none)"]),
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")


# ---------- Full auto mode (API) ----------
def auto_run(repo_root: Path, project: str, objective: str, focus: str = "") -> Path:
    """
    Full pipeline:
    - create run
    - call Manager API -> ticket.json
    - call Dev API -> dev_response.json
    - apply -> writes into projets/<project>/
    """
    p = Progress()

    p.tick("Create run folder")
    run_dir = create_run(repo_root, project, objective, focus=focus)

    p.tick("Load manager prompt")
    manager_prompt = (run_dir / "manager_prompt.txt").read_text(encoding="utf-8")

    p.tick("Manager → API call")
    ticket_obj = generate_json("manager", manager_prompt)

    p.tick("Write ticket.json")
    write_json(run_dir / "ticket.json", ticket_obj)

    p.tick("Build dev prompt")
    ticket = Ticket.model_validate(ticket_obj)
    cfg = load_config(repo_root)
    dev_prompt = make_dev_prompt(cfg, ticket)
    (run_dir / "dev_prompt.txt").write_text(dev_prompt, encoding="utf-8")

    p.tick("Dev → API call")
    dev_obj = generate_json("dev", dev_prompt)

    p.tick("Write dev_response.json")
    write_json(run_dir / "dev_response.json", dev_obj)

    p.tick("Apply file ops into project workspace")
    apply_run(repo_root, run_dir, project=project)

    p.tick("Done")
    return run_dir
