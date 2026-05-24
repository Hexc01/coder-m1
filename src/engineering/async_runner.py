"""Async event stream runner for long-running workflows."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator
from uuid import uuid4

from loguru import logger


class EventType(str, Enum):
    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"
    TOOL_CALL = "tool_call"
    AGENT_MESSAGE = "agent_message"
    CHECKPOINT = "checkpoint"
    ERROR = "error"
    TASK_COMPLETE = "task_complete"


@dataclass
class WorkflowEvent:
    event_type: EventType
    node_name: str
    data: dict
    timestamp: float


class AsyncWorkflowRunner:
    """Runs the LangGraph workflow with async event streaming."""

    def __init__(self, compiled_graph, checkpointer=None):
        self.graph = compiled_graph
        self.checkpointer = checkpointer

    async def run(
        self,
        initial_state: dict,
        thread_id: str | None = None,
    ) -> AsyncIterator[WorkflowEvent]:
        """Execute the graph and yield events as they occur."""
        tid = thread_id or str(uuid4())
        config = {"configurable": {"thread_id": tid}}

        logger.info(f"Starting workflow: thread_id={tid}")

        try:
            async for event in self.graph.astream_events(
                initial_state, config=config, version="v2"
            ):
                kind = event.get("event", "")
                name = event.get("name", "unknown")

                if kind == "on_chain_start":
                    yield WorkflowEvent(EventType.NODE_START, name, event.get("data", {}), time.time())
                elif kind == "on_chain_end":
                    yield WorkflowEvent(EventType.NODE_COMPLETE, name, event.get("data", {}), time.time())
                    yield WorkflowEvent(EventType.CHECKPOINT, name, {"thread_id": tid}, time.time())
                elif kind == "on_tool_start":
                    yield WorkflowEvent(EventType.TOOL_CALL, name, event.get("data", {}), time.time())
                elif kind == "on_tool_end":
                    yield WorkflowEvent(EventType.TOOL_CALL, name, event.get("data", {}), time.time())

        except Exception as e:
            logger.error(f"Workflow error: {e}")
            yield WorkflowEvent(EventType.ERROR, "workflow", {"error": str(e)}, time.time())

        yield WorkflowEvent(EventType.TASK_COMPLETE, "workflow", {"thread_id": tid}, time.time())
        logger.info(f"Workflow complete: thread_id={tid}")

    async def run_sync(self, initial_state: dict, thread_id: str | None = None) -> dict:
        """Run the workflow and return the final state (no streaming)."""
        tid = thread_id or str(uuid4())
        config = {"configurable": {"thread_id": tid}}
        result = await self.graph.ainvoke(initial_state, config=config)
        return result
