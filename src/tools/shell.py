"""Shell command execution tool wrapper."""

from __future__ import annotations

from src.tools.registry import ToolRegistry, ToolResult


async def run_command(registry: ToolRegistry, command: str, timeout: int = 30) -> ToolResult:
    """Execute a shell command via MCP."""
    return await registry.invoke("shell_run_command", {"command": command, "timeout": timeout})


async def run_script(registry: ToolRegistry, script: str, language: str = "bash") -> ToolResult:
    """Execute a script via MCP."""
    return await registry.invoke("shell_run_command", {"command": script})
