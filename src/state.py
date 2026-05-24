from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


# --- Enums ---

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVISION = "needs_revision"


# --- Pydantic Models ---

class SubTask(BaseModel):
    """A single decomposed subtask from the Planner."""
    id: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str = "coder"
    result: str | None = None
    retry_count: int = 0


class CodePatch(BaseModel):
    """A code change produced by the Coder."""
    file_path: str
    old_content: str
    new_content: str
    description: str
    subtask_id: str


class ReviewFeedback(BaseModel):
    """Feedback from the Reviewer agent."""
    subtask_id: str
    verdict: Literal["approve", "revise", "reject"]
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    test_results: dict | None = None


class ToolCallRecord(BaseModel):
    """Record of a tool invocation."""
    tool_name: str
    arguments: dict
    result: str
    success: bool
    duration_ms: float
    subtask_id: str | None = None


class AgentMessage(BaseModel):
    """A structured message between agents."""
    sender: str
    receiver: str
    content: str
    message_type: Literal["task", "feedback", "tool_request", "tool_result", "error"]
    subtask_id: str | None = None
    metadata: dict = Field(default_factory=dict)


# --- LangGraph State ---

class GraphState(TypedDict):
    """The complete state flowing through the LangGraph DAG."""

    # Core task
    task_id: str
    original_request: str
    task_status: TaskStatus

    # Planner outputs
    subtasks: list[SubTask]
    current_subtask_index: int

    # Agent messages (append-only)
    messages: Annotated[list[AgentMessage], operator.add]

    # Coder outputs
    patches: Annotated[list[CodePatch], operator.add]
    current_patch: CodePatch | None

    # Reviewer outputs
    review_feedback: Annotated[list[ReviewFeedback], operator.add]
    latest_review: ReviewFeedback | None

    # Tool usage tracking
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    pending_tool_request: dict | None

    # Memory context (injected from memory system)
    memory_context: str
    similar_past_tasks: list[dict]

    # Control flow
    current_agent: str
    iteration_count: int
    max_iterations: int
    error_log: Annotated[list[str], operator.add]

    # Engineering
    repo_path: str | None
    run_checkpoint_id: str | None


def create_initial_state(
    task_id: str,
    request: str,
    repo_path: str | None = None,
    max_iterations: int = 50,
) -> dict:
    """Create a fresh GraphState for a new task."""
    return {
        "task_id": task_id,
        "original_request": request,
        "task_status": TaskStatus.PENDING,
        "subtasks": [],
        "current_subtask_index": 0,
        "messages": [],
        "patches": [],
        "current_patch": None,
        "review_feedback": [],
        "latest_review": None,
        "tool_calls": [],
        "pending_tool_request": None,
        "memory_context": "",
        "similar_past_tasks": [],
        "current_agent": "",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "error_log": [],
        "repo_path": repo_path,
        "run_checkpoint_id": None,
    }
