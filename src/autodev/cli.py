from __future__ import annotations

import argparse
from pathlib import Path

from .core import create_run, apply_run, auto_run


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
    p_new.add_argument("--focus", default="", help="Optional keyword to pick relevant files")

    # ---- apply ----
    p_apply = sub.add_parser("apply", help="Apply dev_response file_ops for an existing run")
    p_apply.add_argument("--run", required=True, help="Run folder like runs/run_0001")
    p_apply.add_argument("--project", default="", help="Optional override project name")

    # ---- auto ----
    p_auto = sub.add_parser("auto", help="Run Manager+Dev via API, then apply")
    p_auto.add_argument("--project", required=True, help="Project folder name under ./projets/")
    p_auto.add_argument("--objective", default="", help="Objective as a string")
    p_auto.add_argument("--objective-file", default="", help="Path to a text file containing the objective")
    p_auto.add_argument("--focus", default="", help="Optional keyword to pick relevant files")

    args = parser.parse_args()
    repo_root = Path.cwd()

    if args.cmd == "new":
        objective = _load_objective(args)
        run_dir = create_run(repo_root, args.project, objective, focus=args.focus)
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
        run_dir = auto_run(repo_root, args.project, objective, focus=args.focus)
        print(f"Auto completed. Run: {run_dir}")
        print(f"Applied into projets/{args.project}/. See: {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
