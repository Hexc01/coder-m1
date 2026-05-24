from __future__ import annotations

from loguru import logger

from src.agents.base import BaseAgent
from src.llm.prompts import PLANNER_SYSTEM
from src.state import GraphState, SubTask, TaskStatus


class PlannerAgent(BaseAgent):
    """Decomposes user tasks into ordered subtasks."""

    async def execute(self, state: GraphState) -> dict:
        logger.info(f"[Planner] Decomposing task: {state['original_request'][:80]}...")

        memory_context = await self.retriever.query(state["original_request"], n_results=3)
        similar_tasks = await self.retriever.find_similar_tasks(state["original_request"])

        prompt = self._build_planner_prompt(state, memory_context, similar_tasks)
        result = await self.llm.chat_structured(
            messages=[{"role": "user", "content": prompt}],
            system=PLANNER_SYSTEM,
        )

        subtasks = self._parse_subtasks(result)

        logger.info(f"[Planner] Created {len(subtasks)} subtasks")
        return {
            "subtasks": subtasks,
            "current_subtask_index": 0,
            "task_status": TaskStatus.IN_PROGRESS,
            "current_agent": "planner",
            "memory_context": memory_context,
            "similar_past_tasks": similar_tasks,
            "messages": self._emit_message(
                "coder", f"Task decomposed into {len(subtasks)} subtasks", "task"
            ),
        }

    def _build_planner_prompt(
        self, state: GraphState, memory_context: str, similar_tasks: list[dict]
    ) -> str:
        parts = [f"Task: {state['original_request']}"]

        if memory_context:
            parts.append(f"\nRelevant memory context:\n{memory_context}")

        if similar_tasks:
            parts.append("\nSimilar past tasks:")
            for t in similar_tasks[:3]:
                parts.append(f"- {t['document'][:200]}")

        parts.append(
            "\nDecompose this into subtasks. Output a JSON array of objects "
            'with: id, description, dependencies, assigned_agent ("coder" or "tool_agent")'
        )
        return "\n".join(parts)

    def _parse_subtasks(self, result) -> list[SubTask]:
        if isinstance(result, list):
            raw_list = result
        elif isinstance(result, dict):
            raw_list = result.get("subtasks", result.get("tasks", [result]))
        else:
            logger.warning(f"[Planner] Unexpected result type: {type(result)}")
            return [SubTask(id="subtask-1", description=str(result))]

        subtasks = []
        for i, item in enumerate(raw_list):
            try:
                subtasks.append(SubTask(
                    id=item.get("id", f"subtask-{i+1}"),
                    description=item.get("description", str(item)),
                    dependencies=item.get("dependencies", []),
                    assigned_agent=item.get("assigned_agent", "coder"),
                ))
            except Exception as e:
                logger.warning(f"[Planner] Failed to parse subtask {i}: {e}")
        return subtasks or [SubTask(id="subtask-1", description="Complete the task")]
