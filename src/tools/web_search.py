"""Web search tool wrapper (optional)."""

from __future__ import annotations

from src.tools.registry import ToolRegistry, ToolResult


async def web_search(registry: ToolRegistry, query: str, max_results: int = 5) -> ToolResult:
    """Search the web for information."""
    return await registry.invoke("web_search", {"query": query, "max_results": max_results})
