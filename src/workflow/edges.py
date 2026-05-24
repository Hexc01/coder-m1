"""Conditional edge routing logic for the LangGraph workflow."""

from __future__ import annotations

from loguru import logger

from src.config import settings
from src.state import GraphState, TaskStatus


def route_after_planner(state: GraphState) -> str:
    """After planning, route to coder for first subtask."""
    if not state.get("subtasks"):
        logger.info("[Edge] No subtasks, ending")
        return "end"
    first = state["subtasks"][0]
    if first.assigned_agent == "tool_agent":
        return "tool_agent"
    return "coder"


def route_after_review(state: GraphState) -> str:
    """After review: approve -> next_subtask, revise -> coder, reject -> failure."""
    review = state.get("latest_review")
    if review is None:
        return "next_subtask"
    if review.verdict == "approve":
        return "next_subtask"
    elif review.verdict == "revise":
        subtask = state["subtasks"][state["current_subtask_index"]]
        if subtask.retry_count >= settings.max_retries_per_subtask:
            logger.info(f"[Edge] Max retries reached for {subtask.id}")
            return "handle_failure"
        return "coder"
    else:  # reject
        return "handle_failure"


def route_after_subtask(state: GraphState) -> str:
    """After advancing subtask, check if done or continue."""
    if state["task_status"] == TaskStatus.COMPLETED:
        return "memory_reflect"
    if state["task_status"] == TaskStatus.FAILED:
        return "memory_reflect"
    idx = state["current_subtask_index"]
    if idx >= len(state["subtasks"]):
        return "memory_reflect"
    subtask = state["subtasks"][idx]
    if subtask.assigned_agent == "tool_agent":
        return "tool_agent"
    return "coder"


def route_after_failure(state: GraphState) -> str:
    """After failure handling, go to reflection."""
    return "memory_reflect"


def route_after_tool(state: GraphState) -> str:
    """After tool execution, return to the requesting agent."""
    # Check if there's still a pending tool request (shouldn't be, tool_agent clears it)
    if state.get("pending_tool_request"):
        return "tool_agent"
    # Return to the agent that requested the tool, or default to planner
    last_msg = state["messages"][-1] if state.get("messages") else None
    if last_msg and last_msg.sender == "tool_agent":
        receiver = last_msg.receiver
        if receiver in ("coder", "reviewer", "planner"):
            return receiver
    return "planner"


def check_iteration_limit(state: GraphState) -> str:
    """Circuit breaker: check if max iterations exceeded."""
    if state.get("iteration_count", 0) >= state.get("max_iterations", settings.max_iterations):
        logger.warning(f"[Edge] Max iterations ({state['max_iterations']}) reached")
        return "handle_failure"
    return "continue"
