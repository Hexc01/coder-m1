from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from src.agents.base import BaseAgent
from src.llm.prompts import REVIEWER_SYSTEM
from src.state import GraphState, ReviewFeedback


class ReviewerAgent(BaseAgent):
    """Reviews code patches: LLM review + actual test execution."""

    async def execute(self, state: GraphState) -> dict:
        patch = state.get("current_patch")
        if not patch:
            logger.warning("[Reviewer] No patch to review, approving")
            return self._approve("no-patch")

        logger.info(f"[Reviewer] Reviewing patch for {patch.file_path}")

        # Step 1: Write patch to file and run tests
        test_result = self._run_tests(patch, state)

        # Step 2: LLM review with test results included
        prompt = self._build_reviewer_prompt(state, patch, test_result)
        result = await self.llm.chat_structured(
            messages=[{"role": "user", "content": prompt}],
            system=REVIEWER_SYSTEM,
        )

        feedback = self._parse_feedback(result, patch.subtask_id, test_result)

        # If tests failed but LLM approved, override to revise
        if test_result and test_result["exit_code"] != 0 and feedback.verdict == "approve":
            logger.info("[Reviewer] Tests failed but LLM approved, overriding to revise")
            feedback.verdict = "revise"
            if not feedback.issues:
                feedback.issues = ["Tests failed"]

        # Update subtask retry count if revision needed
        subtasks = list(state["subtasks"])
        idx = state["current_subtask_index"]
        if feedback.verdict == "revise":
            subtasks[idx].retry_count += 1

        logger.info(f"[Reviewer] Verdict: {feedback.verdict}, test_exit: {test_result['exit_code'] if test_result else 'N/A'}")
        return {
            "latest_review": feedback,
            "review_feedback": [feedback],
            "subtasks": subtasks,
            "current_agent": "reviewer",
            "messages": self._emit_message(
                "coder" if feedback.verdict == "revise" else "planner",
                f"Review: {feedback.verdict}. Issues: {feedback.issues}. Test output: {test_result['output'][:500] if test_result else 'N/A'}",
                "feedback",
                patch.subtask_id,
            ),
        }

    def _run_tests(self, patch, state: GraphState) -> dict | None:
        """Write patch to disk and run tests. Returns {exit_code, output, duration_ms}."""
        repo_path = state.get("repo_path") or "."
        try:
            # Write the patched file
            file_path = Path(repo_path) / patch.file_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(patch.new_content)

            # Find and run test files
            test_file = self._find_test_file(patch.file_path, repo_path)
            if not test_file:
                # Just do a syntax check
                result = subprocess.run(
                    ["python3", "-c", f"import py_compile; py_compile.compile('{file_path}', doraise=True)"],
                    capture_output=True, text=True, timeout=30, cwd=repo_path,
                )
                return {
                    "exit_code": result.returncode,
                    "output": result.stderr or "Syntax OK",
                    "test_type": "syntax_check",
                }

            # Run pytest on the test file
            result = subprocess.run(
                ["python3", "-m", "pytest", test_file, "-v", "--tb=short"],
                capture_output=True, text=True, timeout=60, cwd=repo_path,
            )
            return {
                "exit_code": result.returncode,
                "output": (result.stdout + result.stderr)[-2000:],
                "test_type": "pytest",
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "output": "Test timed out (60s)", "test_type": "timeout"}
        except Exception as e:
            logger.warning(f"[Reviewer] Test execution error: {e}")
            return None

    def _find_test_file(self, source_file: str, repo_path: str) -> str | None:
        """Find the corresponding test file for a source file."""
        source = Path(source_file)
        candidates = [
            f"test_{source.stem}.py",
            f"tests/test_{source.stem}.py",
            f"test/tests_{source.stem}.py",
        ]
        for candidate in candidates:
            full = Path(repo_path) / candidate
            if full.exists():
                return candidate
        # Search for any test file that might test this module
        test_dir = Path(repo_path) / "tests"
        if test_dir.exists():
            for f in test_dir.glob("test_*.py"):
                return str(f.relative_to(repo_path))
        return None

    def _approve(self, subtask_id: str) -> dict:
        feedback = ReviewFeedback(subtask_id=subtask_id, verdict="approve")
        return {
            "latest_review": feedback,
            "review_feedback": [feedback],
            "current_agent": "reviewer",
        }

    def _build_reviewer_prompt(self, state: GraphState, patch, test_result: dict | None) -> str:
        parts = [
            f"Subtask: {state['subtasks'][state['current_subtask_index']].description}",
            f"\nFile: {patch.file_path}",
            f"Description: {patch.description}",
        ]
        if patch.old_content:
            parts.append(f"\nOld content:\n{patch.old_content[:2000]}")
        parts.append(f"\nNew content:\n{patch.new_content[:2000]}")

        if test_result:
            parts.append(f"\nTest execution ({test_result.get('test_type', 'unknown')}):")
            parts.append(f"Exit code: {test_result['exit_code']}")
            parts.append(f"Output:\n{test_result['output'][:1000]}")
        else:
            parts.append("\nNo test execution performed (syntax check only or skipped).")

        parts.append(
            '\nReview this patch considering the test results. Output JSON: '
            '{"subtask_id": str, "verdict": "approve|revise|reject", '
            '"issues": [str], "suggestions": [str]}'
        )
        return "\n".join(parts)

    def _parse_feedback(self, result, subtask_id: str, test_result: dict | None) -> ReviewFeedback:
        if isinstance(result, dict) and "verdict" in result:
            return ReviewFeedback(
                subtask_id=result.get("subtask_id", subtask_id),
                verdict=result.get("verdict", "approve"),
                issues=result.get("issues", []),
                suggestions=result.get("suggestions", []),
                test_results=test_result,
            )
        logger.warning(f"[Reviewer] Unexpected feedback format: {type(result)}")
        return ReviewFeedback(subtask_id=subtask_id, verdict="approve", test_results=test_result)
