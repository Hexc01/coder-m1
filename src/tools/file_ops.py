"""File system operations tool wrapper."""

from __future__ import annotations

from src.tools.registry import ToolRegistry, ToolResult


async def read_file(registry: ToolRegistry, path: str) -> ToolResult:
    """Read a file's contents."""
    return await registry.invoke("filesystem_read_file", {"path": path})


async def write_file(registry: ToolRegistry, path: str, content: str) -> ToolResult:
    """Write content to a file."""
    return await registry.invoke("filesystem_write_file", {"path": path, "content": content})


async def list_directory(registry: ToolRegistry, path: str = ".") -> ToolResult:
    """List directory contents."""
    return await registry.invoke("filesystem_list_directory", {"path": path})


async def search_files(registry: ToolRegistry, path: str, pattern: str) -> ToolResult:
    """Search for files matching a pattern."""
    return await registry.invoke("filesystem_search_files", {"path": path, "pattern": pattern})
