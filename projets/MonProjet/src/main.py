from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List


@dataclass(frozen=True)
class Question:
    prompt: str
    options: Dict[str, str]
    answer: str


def read_questions_file(path: str | Path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")


def parse_questions(text: str) -> List[Question]:
    # Ignore empty lines and comment lines, then split into blocks with a line exactly equal to '---'.
    blocks: List[List[str]] = [[]]
    for raw in text.splitlines():
        line = raw.rstrip("\n\r")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "---" and line == "---":
            if blocks[-1]:
                blocks.append([])
            continue
        blocks[-1].append(line)

    questions: List[Question] = []
    for block in blocks:
        if not block:
            continue

        prompt: str | None = None
        options: Dict[str, str] = {}
        answer: str | None = None
        valid = True

        for line in block:
            if line.startswith("Q:"):
                prompt = line[2:].strip()
            elif line.startswith("ANS:"):
                cand = line[4:].strip().upper()
                if cand in {"A", "B", "C", "D"}:
                    answer = cand
                else:
                    valid = False
            elif len(line) >= 2 and line[1] == ")" and line[0] in "ABCD":
                key = line[0]
                options[key] = line[2:].strip()
            else:
                # Unknown line in block -> invalid block
                valid = False

        required_keys = {"A", "B", "C", "D"}
        if not valid:
            continue
        if prompt is None or prompt == "":
            continue
        if set(options.keys()) != required_keys:
            continue
        if answer is None:
            continue

        questions.append(Question(prompt=prompt, options=options, answer=answer))

    return questions


def ask_choice(prompt: str) -> str:
    while True:
        try:
            value = input(prompt)
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(0)
        choice = value.strip().upper()
        if choice in {"A", "B", "C", "D"}:
            return choice


def run_quiz(q: Question) -> int:
    print(q.prompt)
    for key in ("A", "B", "C", "D"):
        print(f"{key}) {q.options[key]}")
    user = ask_choice("Your answer (A/B/C/D): ")
    return 1 if user == q.answer else 0


def main() -> int:
    questions_path = Path(__file__).resolve().parent / "questions.txt"
    try:
        text = read_questions_file(questions_path)
    except OSError as e:
        print(f"Error: cannot read questions file: {e}", file=sys.stderr)
        return 1

    questions = parse_questions(text)
    if not questions:
        print("Error: no valid questions found", file=sys.stderr)
        return 1

    score = run_quiz(questions[0])
    print(f"Score: {score}/1")
    return 0 if score == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
