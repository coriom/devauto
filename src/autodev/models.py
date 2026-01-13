from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

Priority = Literal["P0", "P1", "P2"]
FileOpType = Literal["write", "append", "replace"]


class RelevantFile(BaseModel):
    path: str
    excerpt: str


class Ticket(BaseModel):
    schema_version: str = "1.0"
    id: str
    title: str
    goal: str
    priority: Priority = "P1"
    labels: List[str] = Field(default_factory=list)

    # Context (optional but useful)
    relevant_files: List[RelevantFile] = Field(default_factory=list)
    conventions: List[str] = Field(default_factory=list)

    # Planning
    definition_of_done: List[str] = Field(default_factory=list)
    plan_steps: List[str] = Field(default_factory=list)

    # --- New: progress + memory (scalable) ---
    # Where we are now (one paragraph)
    progress_summary: str = ""

    # What remains after this ticket (source-of-truth TODO)
    remaining_work: List[str] = Field(default_factory=list)

    # Partial update to merge into projets/<project>/STATE.json
    # (shallow merge for MVP; can evolve to JSON Patch later)
    state_update: Dict[str, Any] = Field(default_factory=dict)


class FileOp(BaseModel):
    op: FileOpType
    path: str
    content: Optional[str] = None

    # For replace
    find: Optional[str] = None
    replace: Optional[str] = None


class DevResponse(BaseModel):
    schema_version: str = "1.0"
    ticket_id: str
    summary: str
    assumptions: List[str] = Field(default_factory=list)
    file_ops: List[FileOp] = Field(default_factory=list)
    review_notes: List[str] = Field(default_factory=list)
