from __future__ import annotations

import json
import os
from typing import Any, Dict

from openai import OpenAI


def _get_key(role: str) -> str:
    # role in {"manager","dev"}
    if role == "manager":
        return os.environ.get("OPENAI_API_KEY_MANAGER") or os.environ.get("OPENAI_API_KEY") or ""
    if role == "dev":
        return os.environ.get("OPENAI_API_KEY_DEV") or os.environ.get("OPENAI_API_KEY") or ""
    return os.environ.get("OPENAI_API_KEY") or ""


def _get_model(role: str) -> str:
    if role == "manager":
        return os.environ.get("OPENAI_MODEL_MANAGER", "gpt-5.2")
    if role == "dev":
        return os.environ.get("OPENAI_MODEL_DEV", "gpt-5.2")
    return os.environ.get("OPENAI_MODEL_MANAGER", "gpt-5.2")


def generate_json(role: str, prompt_text: str, *, max_retries: int = 2) -> Dict[str, Any]:
    """
    Calls Responses API and expects the model to return ONLY valid JSON.

    Uses JSON mode via:
      text={"format": {"type": "json_object"}}
    so outputs are always valid JSON objects (far fewer repair loops).
    """
    key = _get_key(role)
    if not key:
        raise RuntimeError(
            f"Missing API key for role={role}. Set OPENAI_API_KEY_{role.upper()} in .env (or OPENAI_API_KEY)."
        )

    client = OpenAI(api_key=key)
    model = _get_model(role)

    last_text = ""
    for attempt in range(max_retries + 1):
        user_input = prompt_text if attempt == 0 else (
            "Your previous output was NOT valid JSON.\n"
            "Return ONLY a valid JSON object. No markdown, no commentary.\n\n"
            f"Previous output:\n{last_text}"
        )

        resp = client.responses.create(
            model=model,
            input=user_input,
            # Force JSON object output (JSON mode)
            text={"format": {"type": "json_object"}},
        )

        text = (resp.output_text or "").strip()
        last_text = text

        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                # we asked for json_object; defensive check
                continue
            return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("Model did not return valid JSON after retries.")
