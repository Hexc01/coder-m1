from __future__ import annotations

from loguru import logger

from src.agents.base import BaseAgent
from src.llm.prompts import TOOL_AGENT_SYSTEM
from src.state import GraphState, ToolCallRecord
from src.tools.registry import ToolRegistry


class ToolAgent(BaseAgent):
    """Executes MCP tool calls on behalf of other agents."""

    def __init__(self, *, tool_registry: ToolRegistry, **kwargs):
        super().__init__(**kwargs)
        self.registry = tool_registry

    async def execute(self, state: GraphState) -> dict:
        request = state.get("pending_tool_request")
        if not request:
            logger.warning("[ToolAgent] No pending tool request")
            return {"pending_tool_request": None}

        tool_name = request.get("tool", "")
        arguments = request.get("arguments", {})
        caller = request.get("caller", "unknown")

        logger.info(f"[ToolAgent] Executing '{tool_name}' for {caller}")

        result = await self.registry.invoke(tool_name, arguments)

        record = ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            result=result.output[:2000],
            success=result.success,
            duration_ms=result.duration_ms,
        )

        logger.info(f"[ToolAgent] {'Success' if result.success else 'Failed'} in {result.duration_ms:.0f}ms")
        return {
            "tool_calls": [record],
            "pending_tool_request": None,
            "current_agent": "tool_agent",
            "messages": self._emit_message(
                caller, result.output[:2000], "tool_result"
            ),
        }
