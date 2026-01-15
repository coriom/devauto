from __future__ import annotations

import argparse
from pathlib import Path

# core is now a package: src/autodev/core/__init__.py re-exports these
from .core import apply_run, auto_run, create_run, loop_run


def _load_objective(args) -> str:
    if getattr(args, "objective_file", ""):
        p = Path(args.objective_file)
        if not p.is_file():
            raise SystemExit(f"Objective file not found: {p}")
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit(f"Objective file is empty: {p}")
        return text

    if getattr(args, "objective", ""):
        text = args.objective.strip()
        if not text:
            raise SystemExit("Objective is empty.")
        return text

    raise SystemExit("Provide --objective or --objective-file")


def _load_patch_text(args) -> str:
    """
    Optional patch file that contains prioritized modification instructions.
    Returns empty string if not provided.
    """
    patch_file = getattr(args, "patch_file", "") or ""
    if not patch_file:
        return ""

    p = Path(patch_file)
    if not p.is_file():
        raise SystemExit(f"Patch file not found: {p}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"Patch file is empty: {p}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autodev",
        description="AutoDev (human-in-the-loop + API auto mode)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- new ----
    p_new = sub.add_parser("new", help="Create a new run folder with prompts + templates")
    p_new.add_argument("--project", required=True, help="Project folder name under ./projets/")
    p_new.add_argument("--objective", default="", help="Objective as a string")
    p_new.add_argument("--objective-file", default="", help="Path to a text file containing the objective")
    p_new.add_argument("--patch-file", default="", help="Optional path to a text file containing prioritized patches")
    p_new.add_argument("--focus", default="", help="Optional keyword to pick relevant files")

    # ---- apply ----
    p_apply = sub.add_parser("apply", help="Apply dev_response file_ops for an existing run")
    p_apply.add_argument("--run", required=True, help="Run folder like runs/run_0001")
    p_apply.add_argument("--project", default="", help="Optional override project name")

    # ---- auto ----
    p_auto = sub.add_parser("auto", help="Run Manager+Dev via API, then apply (single iteration)")
    p_auto.add_argument("--project", required=True, help="Project folder name under ./projets/")
    p_auto.add_argument("--objective", default="", help="Objective as a string")
    p_auto.add_argument("--objective-file", default="", help="Path to a text file containing the objective")
    p_auto.add_argument("--patch-file", default="", help="Optional path to a text file containing prioritized patches")
    p_auto.add_argument("--focus", default="", help="Optional keyword to pick relevant files")

    # ---- loop ----
    p_loop = sub.add_parser("loop", help="Run multiple auto iterations (stop when todo empty or max reached)")
    p_loop.add_argument("--project", required=True, help="Project folder name under ./projets/")
    p_loop.add_argument("--objective", default="", help="Objective as a string")
    p_loop.add_argument("--objective-file", default="", help="Path to a text file containing the objective")
    p_loop.add_argument("--patch-file", default="", help="Optional path to a text file containing prioritized patches")
    p_loop.add_argument("--focus", default="", help="Optional keyword to pick relevant files")
    p_loop.add_argument("--max-iter", type=int, default=10, help="Hard cap on iterations")
    p_loop.add_argument(
        "--no-stop-on-empty-todo",
        action="store_true",
        help="Do not stop when remaining_work/todo becomes empty (rarely useful)",
    )
    p_loop.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=2,
        help="Stop loop after this many consecutive errors (circuit breaker)",
    )

    args = parser.parse_args()
    repo_root = Path.cwd()

    if args.cmd == "new":
        objective = _load_objective(args)
        patch_text = _load_patch_text(args)

        # Backward-compatible: only pass patch_text if core supports it.
        # (Once core is updated, it should accept patch_text / patch_file.)
        try:
            run_dir = create_run(
                repo_root,
                args.project,
                objective,
                focus=args.focus,
                objective_file=args.objective_file,
                patch_text=patch_text,
                patch_file=args.patch_file,
            )
        except TypeError:
            # Core not updated yet; fall back to old signature
            run_dir = create_run(
                repo_root,
                args.project,
                objective,
                focus=args.focus,
                objective_file=args.objective_file,
            )

        print(f"Created: {run_dir}")
        print(f"- Open {run_dir / 'manager_prompt.txt'} and paste Manager JSON into {run_dir / 'ticket.json'}")
        print(f"- Ask DEV with the Ticket JSON, paste response into {run_dir / 'dev_response.json'}")
        print(f"- Then run: autodev apply --run {run_dir.as_posix()}")

    elif args.cmd == "apply":
        run_dir = Path(args.run)
        if not run_dir.is_dir():
            raise SystemExit(f"Run dir not found: {run_dir}")

        project = args.project.strip() or None
        apply_run(repo_root, run_dir, project=project)
        print(f"Applied ops. See: {run_dir / 'summary.md'}")

    elif args.cmd == "auto":
        objective = _load_objective(args)
        patch_text = _load_patch_text(args)

        try:
            run_dir = auto_run(
                repo_root,
                args.project,
                objective,
                focus=args.focus,
                objective_file=args.objective_file,
                patch_text=patch_text,
                patch_file=args.patch_file,
            )
        except TypeError:
            run_dir = auto_run(
                repo_root,
                args.project,
                objective,
                focus=args.focus,
                objective_file=args.objective_file,
            )

        print(f"Auto completed. Run: {run_dir}")
        print(f"Applied into projets/{args.project}/. See: {run_dir / 'summary.md'}")

    elif args.cmd == "loop":
        objective = _load_objective(args)
        patch_text = _load_patch_text(args)
        stop_on_empty = not args.no_stop_on_empty_todo

        try:
            summary = loop_run(
                repo_root,
                args.project,
                objective,
                objective_file=args.objective_file,
                focus=args.focus,
                max_iter=args.max_iter,
                stop_on_empty_todo=stop_on_empty,
                max_consecutive_errors=args.max_consecutive_errors,
                patch_text=patch_text,
                patch_file=args.patch_file,
            )
        except TypeError:
            summary = loop_run(
                repo_root,
                args.project,
                objective,
                objective_file=args.objective_file,
                focus=args.focus,
                max_iter=args.max_iter,
                stop_on_empty_todo=stop_on_empty,
                max_consecutive_errors=args.max_consecutive_errors,
            )

        print("\nLoop summary:")
        print(f"- Project: {summary.get('project')}")
        print(f"- State ID: {summary.get('state_id')}")
        print(f"- Iterations run: {summary.get('iterations_run')}")
        print(f"- Done: {summary.get('done')}")
        print(f"- Remaining todo_len: {summary.get('todo_len')}")
        print(f"- Last run: {summary.get('last_run')}")
        runs = summary.get("runs") or []
        if runs:
            print(f"- Runs: {', '.join(runs)}")

    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
