from __future__ import annotations

from typing import List, Literal, Optional
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

    relevant_files: List[RelevantFile] = Field(default_factory=list)
    conventions: List[str] = Field(default_factory=list)

    definition_of_done: List[str] = Field(default_factory=list)
    plan_steps: List[str] = Field(default_factory=list)

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
