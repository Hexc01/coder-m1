"""Git operations tool wrapper."""

from __future__ import annotations

from src.tools.registry import ToolRegistry, ToolResult


async def git_status(registry: ToolRegistry, repo_path: str = ".") -> ToolResult:
    """Get git status."""
    return await registry.invoke("git_status", {"repo_path": repo_path})


async def git_diff(registry: ToolRegistry, repo_path: str = ".", staged: bool = False) -> ToolResult:
    """Get git diff."""
    return await registry.invoke("git_diff", {"repo_path": repo_path, "staged": staged})


async def git_commit(registry: ToolRegistry, message: str, repo_path: str = ".") -> ToolResult:
    """Create a git commit."""
    return await registry.invoke("git_commit", {"message": message, "repo_path": repo_path})


async def git_log(registry: ToolRegistry, repo_path: str = ".", count: int = 10) -> ToolResult:
    """Get git log."""
    return await registry.invoke("git_log", {"repo_path": repo_path, "count": count})
