"""Git-based incremental patch commit and auto rollback."""

from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger


class PatchManager:
    """Manages incremental git commits and rollback for code patches."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self._commit_stack: list[str] = []

    def apply_patch(self, file_path: str, new_content: str, message: str) -> str:
        """Apply a patch and create a git commit."""
        full_path = self.repo_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(new_content)
        self._run_git("add", file_path)
        self._run_git("commit", "-m", message)
        commit_hash = self._run_git("rev-parse", "HEAD").strip()
        self._commit_stack.append(commit_hash)
        logger.info(f"Applied patch: {file_path} -> {commit_hash[:8]}")
        return commit_hash

    def rollback_last(self) -> bool:
        """Rollback the last patch commit."""
        if not self._commit_stack:
            return False
        commit = self._commit_stack.pop()
        try:
            self._run_git("revert", "--no-edit", commit)
            logger.info(f"Rolled back commit: {commit[:8]}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def rollback_to(self, commit_hash: str) -> bool:
        """Rollback to a specific commit."""
        try:
            self._run_git("reset", "--hard", commit_hash)
            logger.info(f"Reset to commit: {commit_hash[:8]}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Reset failed: {e}")
            return False

    def _run_git(self, *args) -> str:
        """Run a git command in the repo directory."""
        result = subprocess.run(
            ["git"] + list(args),
            cwd=self.repo_path,
            capture_output=True, text=True, check=True,
        )
        return result.stdout
