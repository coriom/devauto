from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import List, Optional


VALID_ANSWERS = ("A", "B", "C", "D")
MAX_INPUT_LEN = 16


@dataclass(frozen=True)
class Question:
    prompt: str
    a: str
    b: str
    c: str
    d: str
    answer: str  # "A"|"B"|"C"|"D"


class QuizFormatError(ValueError):
    pass


def _script_dir() -> Path:
    # Works whether launched as `python src/main.py` or module execution.
    return Path(__file__).resolve().parent


def questions_file_path() -> Path:
    return _script_dir() / "questions.txt"


def _split_blocks(text: str) -> List[List[str]]:
    # Split on one or more blank lines. Keep only non-empty stripped lines.
    raw_blocks = re.split(r"\n\s*\n+", text.strip(), flags=re.MULTILINE)
    blocks: List[List[str]] = []
    for b in raw_blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if lines:
            blocks.append(lines)
    return blocks


def _strip_choice_prefix(line: str, expected_letter: str) -> str:
    # Accept: "A)", "A.", "A:", "A -", "A " etc.
    pattern = rf"^\s*{re.escape(expected_letter)}\s*[\)\.:\-]?\s*"
    return re.sub(pattern, "", line, count=1).strip()


def _parse_block(lines: List[str], idx: int) -> Question:
    if len(lines) != 6:
        raise QuizFormatError(
            f"Format invalide: le bloc #{idx} doit contenir 6 lignes (question, A, B, C, D, réponse). Trouvé: {len(lines)} ligne(s)."
        )

    prompt = lines[0].strip()
    if not prompt:
        raise QuizFormatError(f"Format invalide: la question du bloc #{idx} est vide.")

    a = _strip_choice_prefix(lines[1], "A")
    b = _strip_choice_prefix(lines[2], "B")
    c = _strip_choice_prefix(lines[3], "C")
    d = _strip_choice_prefix(lines[4], "D")

    if not all([a, b, c, d]):
        raise QuizFormatError(f"Format invalide: un ou plusieurs choix vides dans le bloc #{idx}.")

    ans = lines[5].strip().upper()
    # If answer line contains extra text, try to capture first A-D.
    m = re.search(r"\b([ABCD])\b", ans)
    if m:
        ans = m.group(1)

    if ans not in VALID_ANSWERS:
        raise QuizFormatError(
            f"Format invalide: la réponse du bloc #{idx} doit être une lettre parmi A/B/C/D. Trouvé: {lines[5]!r}."
        )

    return Question(prompt=prompt, a=a, b=b, c=c, d=d, answer=ans)


def load_questions(path: Path) -> List[Question]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Fichier introuvable: {path}") from e
    except OSError as e:
        raise OSError(f"Impossible de lire le fichier: {path} ({e})") from e

    blocks = _split_blocks(text)
    if not blocks:
        raise QuizFormatError("Format invalide: aucune question trouvée dans le fichier.")

    questions: List[Question] = []
    for i, lines in enumerate(blocks, start=1):
        questions.append(_parse_block(lines, i))

    return questions


def read_answer(prompt: str = "Votre réponse (A/B/C/D): ") -> str:
    while True:
        try:
            raw = input(prompt)
        except EOFError:
            # Treat as user cancellation.
            print("\nEntrée interrompue. Sortie.")
            raise SystemExit(1)
        except KeyboardInterrupt:
            print("\nInterrompu. Sortie.")
            raise SystemExit(1)

        if raw is None:
            continue

        s = raw.strip().upper()
        if not s:
            print("Entrée invalide: réponse vide. Veuillez entrer A, B, C ou D.")
            continue
        if len(s) > MAX_INPUT_LEN:
            print("Entrée invalide: trop longue. Veuillez entrer uniquement A, B, C ou D.")
            continue
        if s not in VALID_ANSWERS:
            print("Entrée invalide: veuillez entrer A, B, C ou D.")
            continue
        return s


def run_quiz(questions: List[Question]) -> int:
    score = 0

    for i, q in enumerate(questions, start=1):
        print(f"\nQuestion {i}/{len(questions)}")
        print(q.prompt)
        print(f"A) {q.a}")
        print(f"B) {q.b}")
        print(f"C) {q.c}")
        print(f"D) {q.d}")

        ans = read_answer()
        if ans == q.answer:
            score += 1

    return score


def main(argv: Optional[List[str]] = None) -> int:
    path = questions_file_path()
    try:
        questions = load_questions(path)
    except (FileNotFoundError, OSError, QuizFormatError) as e:
        print(str(e))
        return 1

    try:
        score = run_quiz(questions)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

    print(f"\nScore: {score}/{len(questions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
