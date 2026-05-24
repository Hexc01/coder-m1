from __future__ import annotations

import json

from loguru import logger

from src.llm.client import ClaudeClient
from src.memory.long_term import LongTermMemory
from src.state import GraphState, TaskStatus


class MemoryReflector:
    """Post-task reflection: summarize outcomes and store learnings."""

    def __init__(self, llm: ClaudeClient, ltm: LongTermMemory):
        self.llm = ltm
        self._llm = llm
        self._ltm = ltm

    async def reflect(self, state: GraphState) -> None:
        """Analyze completed task and store reflections in long-term memory."""
        try:
            prompt = self._build_reflection_prompt(state)
            result = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                system=(
                    "You are analyzing a completed software engineering task. "
                    "Summarize what worked, what failed, and key lessons. "
                    "Respond in JSON: {\"summary\": str, \"lessons\": [str], \"patterns\": [str]}"
                ),
            )
            reflection = self._parse_reflection(result["content"])
            success = state["task_status"] == TaskStatus.COMPLETED

            self._ltm.store_reflection(
                task_summary=reflection.get("summary", state["original_request"][:200]),
                success=success,
                lessons=reflection.get("lessons", []),
            )
            logger.info(f"Stored reflection for task {state['task_id']}: success={success}")

        except Exception as e:
            logger.error(f"Reflection failed: {e}")

        self._store_errors(state)
        if state["task_status"] == TaskStatus.COMPLETED:
            self._store_patches(state)

    def _store_errors(self, state: GraphState) -> None:
        """Store error patterns from the task."""
        for error in state.get("error_log", []):
            try:
                self._ltm.store_error(
                    error_type="task_error",
                    error_msg=error[:500],
                    fix="(no fix recorded)",
                    context=state["original_request"][:200],
                )
            except Exception as e:
                logger.warning(f"Failed to store error: {e}")

    def _store_patches(self, state: GraphState) -> None:
        """Store successful patches from the task."""
        for patch in state.get("patches", []):
            try:
                self._ltm.store_patch(
                    file_path=patch.file_path,
                    description=patch.description,
                    patch_content=patch.new_content[:2000],
                )
            except Exception as e:
                logger.warning(f"Failed to store patch: {e}")

    def _build_reflection_prompt(self, state: GraphState) -> str:
        """Build the reflection prompt from task state."""
        return f"""Analyze this completed software engineering task:

Task: {state['original_request']}
Status: {state['task_status'].value}
Subtasks: {len(state.get('subtasks', []))}
Iterations: {state.get('iteration_count', 0)}
Errors: {state.get('error_log', [])}
Tool calls made: {len(state.get('tool_calls', []))}
Patches generated: {len(state.get('patches', []))}

Respond with JSON:
{{"summary": "one paragraph summary of what was done", "lessons": ["lesson1", "lesson2"], "patterns": ["pattern1"]}}"""

    def _parse_reflection(self, content: str) -> dict:
        """Parse the reflection JSON response."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"summary": content[:500], "lessons": [], "patterns": []}
