from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def norm_rel_path(p: str) -> Optional[str]:
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
        p = norm_rel_path(raw)
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

    files = raw_ticket.get("files_to_modify")
    if isinstance(files, list):
        out: List[str] = []
        seen = set()
        for x in files:
            if not isinstance(x, str):
                continue
            p = norm_rel_path(x)
            if not p:
                continue
            k = p.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
        if out:
            return out

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

    return extract_paths_from_text("\n".join(texts))


def extract_paths_from_remaining_work(items: List[str]) -> List[str]:
    if not isinstance(items, list):
        return []
    blob = "\n".join([x for x in items if isinstance(x, str)])
    return extract_paths_from_text(blob)
