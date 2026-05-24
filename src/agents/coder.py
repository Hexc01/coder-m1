from __future__ import annotations

from loguru import logger

from src.agents.base import BaseAgent
from src.llm.prompts import CODER_SYSTEM
from src.state import CodePatch, GraphState


class CoderAgent(BaseAgent):
    """Generates code patches for subtasks."""

    async def execute(self, state: GraphState) -> dict:
        idx = state["current_subtask_index"]
        subtask = state["subtasks"][idx]
        logger.info(f"[Coder] Working on subtask: {subtask.id} — {subtask.description[:60]}")

        code_context = await self.retriever.query(subtask.description, n_results=5)
        relevant_patches = await self.retriever.find_relevant_patches(subtask.description)

        prompt = self._build_coder_prompt(state, subtask.description, code_context, relevant_patches)
        result = await self.llm.chat_structured(
            messages=[{"role": "user", "content": prompt}],
            system=CODER_SYSTEM,
        )

        patch = self._parse_patch(result, subtask.id)

        # Check if a tool request was embedded in the response
        pending_tool = None
        if isinstance(result, dict) and "tool" in result:
            pending_tool = {"tool": result["tool"], "arguments": result.get("arguments", {}), "caller": "coder"}

        logger.info(f"[Coder] Generated patch for {patch.file_path}")
        updates: dict = {
            "current_patch": patch,
            "current_agent": "coder",
            "messages": self._emit_message(
                "reviewer", f"Patch ready for {patch.file_path}: {patch.description}", "task", subtask.id
            ),
        }
        if pending_tool:
            updates["pending_tool_request"] = pending_tool
        return updates

    def _build_coder_prompt(
        self, state: GraphState, description: str, code_context: str, relevant_patches: list[dict]
    ) -> str:
        parts = [f"Subtask: {description}"]

        if state.get("repo_path"):
            parts.append(f"Repository: {state['repo_path']}")

        if code_context:
            parts.append(f"\nRelevant context from memory:\n{code_context}")

        if relevant_patches:
            parts.append("\nSimilar successful patches:")
            for p in relevant_patches[:2]:
                parts.append(f"- {p['document'][:300]}")

        if state.get("latest_review"):
            review = state["latest_review"]
            if review.issues:
                parts.append(f"\nPrevious review issues to fix: {review.issues}")

        parts.append(
            '\nGenerate a code patch. Output JSON: {"file_path": str, "old_content": str, '
            '"new_content": str, "description": str}. '
            "For new files, old_content should be empty string."
        )
        return "\n".join(parts)

    def _parse_patch(self, result, subtask_id: str) -> CodePatch:
        if isinstance(result, dict) and "file_path" in result:
            return CodePatch(
                file_path=result["file_path"],
                old_content=result.get("old_content", ""),
                new_content=result.get("new_content", ""),
                description=result.get("description", "Code change"),
                subtask_id=subtask_id,
            )
        logger.warning(f"[Coder] Unexpected patch format, wrapping as string: {type(result)}")
        return CodePatch(
            file_path="output.py",
            old_content="",
            new_content=str(result),
            description="Generated code",
            subtask_id=subtask_id,
        )
