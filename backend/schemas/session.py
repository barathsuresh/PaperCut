from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypedDict

from backend.schemas.contract import ArchitectureBlueprint


class SessionHistoryEntry(TypedDict):
    role: Literal["user", "assistant"]
    text: str


class Node0StoredResult(TypedDict):
    result: str
    reason: str


class Node1StoredResult(TypedDict, total=False):
    model_type: str
    architecture: dict[str, Any]
    objective: str
    key_operations: list[str]
    math_notes: str


class Node2StoredResult(TypedDict, total=False):
    status: str
    output_dir: str
    files: dict[str, int]
    uploaded_files: list[str]
    error: str
    error_type: str


class Node3StoredResult(TypedDict, total=False):
    status: str
    output_dir: str
    stub_files: list[str]
    bottlenecks: list[dict[str, Any]]
    meta: dict[str, Any]
    uploaded_files: list[str]
    error: str
    error_type: str


class SessionData(TypedDict, total=False):
    gcs_uri: str
    name: str
    uploaded_at: str
    history: list[SessionHistoryEntry]
    node0_result: Node0StoredResult
    node1_result: Node1StoredResult
    node2_result: Node2StoredResult
    node3_result: Node3StoredResult


SessionStore = dict[str, SessionData]
