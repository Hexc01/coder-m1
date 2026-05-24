from __future__ import annotations

from loguru import logger

from src.agents.base import BaseAgent
from src.llm.prompts import REVIEWER_SYSTEM
from src.state import GraphState, ReviewFeedback


class ReviewerAgent(BaseAgent):
    """Reviews code patches and provides structured feedback."""

    async def execute(self, state: GraphState) -> dict:
        patch = state.get("current_patch")
        if not patch:
            logger.warning("[Reviewer] No patch to review, approving")
            return self._approve("no-patch")

        logger.info(f"[Reviewer] Reviewing patch for {patch.file_path}")

        prompt = self._build_reviewer_prompt(state, patch)
        result = await self.llm.chat_structured(
            messages=[{"role": "user", "content": prompt}],
            system=REVIEWER_SYSTEM,
        )

        feedback = self._parse_feedback(result, patch.subtask_id)

        # Update subtask retry count if revision needed
        subtasks = list(state["subtasks"])
        idx = state["current_subtask_index"]
        if feedback.verdict == "revise":
            subtasks[idx].retry_count += 1

        logger.info(f"[Reviewer] Verdict: {feedback.verdict}")
        return {
            "latest_review": feedback,
            "review_feedback": [feedback],
            "subtasks": subtasks,
            "current_agent": "reviewer",
            "messages": self._emit_message(
                "coder" if feedback.verdict == "revise" else "planner",
                f"Review: {feedback.verdict}. Issues: {feedback.issues}",
                "feedback",
                patch.subtask_id,
            ),
        }

    def _approve(self, subtask_id: str) -> dict:
        from src.state import TaskStatus
        feedback = ReviewFeedback(subtask_id=subtask_id, verdict="approve")
        return {
            "latest_review": feedback,
            "review_feedback": [feedback],
            "current_agent": "reviewer",
        }

    def _build_reviewer_prompt(self, state: GraphState, patch) -> str:
        parts = [
            f"Subtask: {state['subtasks'][state['current_subtask_index']].description}",
            f"\nFile: {patch.file_path}",
            f"Description: {patch.description}",
        ]
        if patch.old_content:
            parts.append(f"\nOld content:\n{patch.old_content[:2000]}")
        parts.append(f"\nNew content:\n{patch.new_content[:2000]}")
        parts.append(
            '\nReview this patch. Output JSON: {"subtask_id": str, "verdict": "approve|revise|reject", '
            '"issues": [str], "suggestions": [str]}'
        )
        return "\n".join(parts)

    def _parse_feedback(self, result, subtask_id: str) -> ReviewFeedback:
        if isinstance(result, dict) and "verdict" in result:
            return ReviewFeedback(
                subtask_id=result.get("subtask_id", subtask_id),
                verdict=result.get("verdict", "approve"),
                issues=result.get("issues", []),
                suggestions=result.get("suggestions", []),
            )
        logger.warning(f"[Reviewer] Unexpected feedback format: {type(result)}")
        return ReviewFeedback(subtask_id=subtask_id, verdict="approve")
