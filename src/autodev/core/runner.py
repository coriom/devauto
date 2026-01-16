from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ..config import load_config
from ..io import ensure_dir
from ..llm import generate_json
from ..models import DevResponse, Ticket
from ..progress import Progress

from .constants import MAX_DEV_RETRIES, MAX_MANAGER_RETRIES, RUNS_DIRNAME
from .context import build_context, make_dev_prompt, make_manager_prompt
from .logging import log_event
from .ops import apply_file_ops
from .state import (
    load_or_init_state,
    save_state,
    state_id_from_objective,
    states_dir,
    update_state_files_after_ops,
)
from .utils import read_json, sanitize_remaining_work, utc_now_iso, write_json
from .validation import (
    manager_validation_errors,
    repair_prompt,
    validate_dev_ops_nonempty,
    validate_dev_ops_scope,
)


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


def get_workspace_root(repo_root: Path, project: str) -> Path:
    cfg = load_config(repo_root)
    project = (project or "").strip()
    if not project:
        raise ValueError("Project name is required (e.g. --project DeOpt)")
    return cfg.projects_root / project


def create_run(
    repo_root: Path,
    project: str,
    objective: str,
    focus: str = "",
    objective_file: str = "",
    *,
    patch_text: str = "",
    patch_file: str = "",
    mode: str = "human",  # "human" or "auto"
) -> Path:
    cfg = load_config(repo_root)
    workspace_root = get_workspace_root(repo_root, project)
    ensure_dir(workspace_root)
    ensure_dir(states_dir(workspace_root))

    run_dir = next_run_dir(repo_root)
    ensure_dir(run_dir)

    st = load_or_init_state(workspace_root, objective=objective, objective_file=objective_file)

    conventions, relevant_files = build_context(
        workspace_root,
        cfg,
        focus=focus,
        objective_text=objective,
    )
    manager_prompt = make_manager_prompt(
        cfg,
        objective,
        conventions,
        relevant_files,
        state=st,
        patch_text=patch_text or "",
        patch_file=patch_file or "",
    )

    # IMPORTANT: always persist objective (+ patch info) into run_meta.json
    meta = {
        "ts": utc_now_iso(),
        "project": project,
        "workspace_root": workspace_root.as_posix(),
        "focus": focus,
        "objective": objective,
        "objective_file": objective_file or None,
        "patch_file": patch_file or None,
        "patch_text": patch_text or "",
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
    project_name = (project or meta.get("project") or "").strip()
    if not project_name:
        raise ValueError("Project not found. Provide --project or ensure run_meta.json contains it.")

    workspace_root = get_workspace_root(repo_root, project_name)

    objective_text = meta.get("objective", "") or ""
    objective_file = meta.get("objective_file") or ""
    st = load_or_init_state(workspace_root, objective=objective_text, objective_file=objective_file)

    raw_ticket = read_json(run_dir / "ticket.json")
    try:
        ticket = Ticket.model_validate(raw_ticket)
    except ValidationError as e:
        log_event(
            repo_root,
            {"ts": utc_now_iso(), "type": "apply_error", "run": run_dir.name, "error": f"Invalid ticket.json: {e}"},
        )
        raise ValueError(f"Invalid ticket.json: {e}") from e

    # Strict required fields (runtime checks)
    if "progress_summary" not in raw_ticket or not str(raw_ticket.get("progress_summary", "")).strip():
        raise ValueError("Manager ticket must include a non-empty progress_summary.")
    if "remaining_work" not in raw_ticket or not isinstance(raw_ticket.get("remaining_work"), list):
        raise ValueError("Manager ticket must include remaining_work as a list (may be empty when completed).")

    # Mode B REQUIRED (now enforced, since it's in the model + prompts)
    if not isinstance(raw_ticket.get("files_to_modify"), list) or len(raw_ticket.get("files_to_modify")) == 0:
        raise ValueError("Manager ticket must include files_to_modify as a non-empty list (Mode B).")
    if not isinstance(raw_ticket.get("rationale_by_file"), dict) or len(raw_ticket.get("rationale_by_file")) == 0:
        raise ValueError("Manager ticket must include rationale_by_file as a non-empty object (Mode B).")

    raw_dev = read_json(run_dir / "dev_response.json")
    try:
        dev = DevResponse.model_validate(raw_dev)
    except ValidationError as e:
        log_event(
            repo_root,
            {
                "ts": utc_now_iso(),
                "type": "apply_error",
                "run": run_dir.name,
                "error": f"Invalid dev_response.json: {e}",
            },
        )
        raise ValueError(f"Invalid dev_response.json: {e}") from e

    if dev.ticket_id != ticket.id:
        raise ValueError("dev_response.ticket_id does not match ticket.id")

    # --- Mode B hard gate: DEV may only touch Ticket.files_to_modify (+ pinned rule) ---
    scope_errs = validate_dev_ops_scope(raw_dev, raw_ticket, st=st)
    if scope_errs:
        log_event(
            repo_root,
            {
                "ts": utc_now_iso(),
                "type": "apply_error",
                "run": run_dir.name,
                "error": "Dev ops out of scope: " + " | ".join(scope_errs),
            },
        )
        raise ValueError("Dev ops out of scope: " + " | ".join(scope_errs))

    # Apply ops
    apply_file_ops(workspace_root, cfg, dev.file_ops)

    # Update state
    remaining = sanitize_remaining_work(ticket.remaining_work)
    is_complete = len(remaining) == 0

    st_after = dict(st)
    st_after["last_run"] = run_dir.name
    st_after["progress_summary"] = ticket.progress_summary
    st_after["todo"] = remaining

    if is_complete:
        st_after["phase"] = "completed"
    else:
        if st_after.get("phase") == "completed":
            st_after["phase"] = "implementation"

    # Shallow merge state_update
    if isinstance(ticket.state_update, dict) and ticket.state_update:
        for k, v in ticket.state_update.items():
            st_after[k] = v

    # Mode B: file tracking + pinning
    touched_paths = [op.path.replace("\\", "/") for op in dev.file_ops]
    pin_files = raw_ticket.get("pin_files", [])
    if not isinstance(pin_files, list):
        pin_files = []

    update_state_files_after_ops(
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


def auto_run(
    repo_root: Path,
    project: str,
    objective: str,
    focus: str = "",
    objective_file: str = "",
    *,
    patch_text: str = "",
    patch_file: str = "",
) -> Path:
    p = Progress()

    p.tick("Create run folder (+ load objective state)")
    run_dir = create_run(
        repo_root,
        project,
        objective,
        focus=focus,
        objective_file=objective_file,
        patch_text=patch_text or "",
        patch_file=patch_file or "",
        mode="auto",
    )

    cfg = load_config(repo_root)
    meta = read_json(run_dir / "run_meta.json")
    workspace_root = Path(meta["workspace_root"])
    st = load_or_init_state(workspace_root, objective=objective, objective_file=objective_file)

    p.tick("Build manager prompt (in-memory)")
    conventions, relevant_files = build_context(workspace_root, cfg, focus=focus, objective_text=objective)
    base_manager_prompt = make_manager_prompt(
        cfg,
        objective,
        conventions,
        relevant_files,
        state=st,
        patch_text=patch_text or meta.get("patch_text", "") or "",
        patch_file=patch_file or meta.get("patch_file", "") or "",
    )

    # -------- Manager (with retries) --------
    p.tick("Manager → API call")
    manager_prompt = base_manager_prompt
    ticket_obj: Dict[str, Any] = {}

    for attempt in range(0, MAX_MANAGER_RETRIES + 1):
        ticket_obj = generate_json("manager", manager_prompt)

        log_event(
            repo_root,
            {"ts": utc_now_iso(), "type": "manager_ticket", "run": run_dir.name, "attempt": attempt, "ticket": ticket_obj},
        )

        errs = manager_validation_errors(ticket_obj, st)

        # Validate full Ticket schema
        try:
            _ = Ticket.model_validate(ticket_obj)
        except ValidationError as e:
            errs.append(f"Ticket schema validation error: {e}")

        if not errs:
            break

        log_event(
            repo_root,
            {"ts": utc_now_iso(), "type": "manager_ticket_invalid", "run": run_dir.name, "attempt": attempt, "errors": errs},
        )

        if attempt >= MAX_MANAGER_RETRIES:
            raise ValueError("Manager ticket invalid: " + " | ".join(errs))

        manager_prompt = repair_prompt(base_manager_prompt, errs, ticket_obj)

    ticket_obj["remaining_work"] = sanitize_remaining_work(ticket_obj.get("remaining_work", []))

    p.tick("Write ticket.json")
    write_json(run_dir / "ticket.json", ticket_obj)

    ticket = Ticket.model_validate(ticket_obj)

    # -------- Dev (with retries) --------
    p.tick("Build dev prompt (in-memory)")
    base_dev_prompt = make_dev_prompt(cfg, ticket)
    dev_prompt = base_dev_prompt

    p.tick("Dev → API call")
    dev_obj: Dict[str, Any] = {}
    for attempt in range(0, MAX_DEV_RETRIES + 1):
        dev_obj = generate_json("dev", dev_prompt)

        log_event(
            repo_root,
            {"ts": utc_now_iso(), "type": "dev_response", "run": run_dir.name, "attempt": attempt, "dev_response": dev_obj},
        )

        errs = validate_dev_ops_nonempty(dev_obj)

        # --- Mode B hard gate (pre-pydantic): scope + pinned enforcement ---
        if not errs:
            errs.extend(validate_dev_ops_scope(dev_obj, ticket_obj, st=st))

        if not errs:
            try:
                _ = DevResponse.model_validate(dev_obj)
                break
            except ValidationError as e:
                errs.append(f"Invalid DevResponse schema: {e}")

        log_event(
            repo_root,
            {"ts": utc_now_iso(), "type": "dev_response_invalid", "run": run_dir.name, "attempt": attempt, "errors": errs},
        )

        if attempt >= MAX_DEV_RETRIES:
            raise ValueError("Dev response invalid: " + " | ".join(errs))

        dev_prompt = repair_prompt(base_dev_prompt, errs, dev_obj)

    p.tick("Write dev_response.json")
    write_json(run_dir / "dev_response.json", dev_obj)

    # -------- Apply --------
    p.tick("Apply file ops + update objective state")
    try:
        meta["objective"] = objective
        meta["objective_file"] = objective_file or None
        meta["patch_file"] = patch_file or meta.get("patch_file") or None
        meta["patch_text"] = patch_text or meta.get("patch_text", "") or ""
        write_json(run_dir / "run_meta.json", meta)

        apply_run(repo_root, run_dir, project=project)
    except Exception as e:
        log_event(repo_root, {"ts": utc_now_iso(), "type": "apply_error", "run": run_dir.name, "error": str(e)})
        raise

    p.tick("Done")
    return run_dir


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
    patch_text: str = "",
    patch_file: str = "",
) -> Dict[str, Any]:
    workspace_root = get_workspace_root(repo_root, project)
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

        if stop_on_empty_todo and isinstance(todo, list) and len(todo) == 0 and st.get("last_run") is not None:
            log_event(repo_root, {"ts": utc_now_iso(), "type": "loop_stop_done", "project": project, "state_id": sid, "iter": i})
            break

        log_event(
            repo_root,
            {"ts": utc_now_iso(), "type": "loop_iter_start", "project": project, "state_id": sid, "iter": i, "todo_len": len(todo) if isinstance(todo, list) else None},
        )

        try:
            rd = auto_run(
                repo_root,
                project,
                objective,
                focus=focus,
                objective_file=objective_file,
                patch_text=patch_text or "",
                patch_file=patch_file or "",
            )
            runs.append(rd.name)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            log_event(
                repo_root,
                {"ts": utc_now_iso(), "type": "loop_iter_error", "project": project, "state_id": sid, "iter": i, "error": str(e), "consecutive_errors": consecutive_errors},
            )
            if consecutive_errors >= max_consecutive_errors:
                log_event(
                    repo_root,
                    {"ts": utc_now_iso(), "type": "loop_stop_errors", "project": project, "state_id": sid, "iter": i, "consecutive_errors": consecutive_errors},
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
