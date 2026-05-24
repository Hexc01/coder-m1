from __future__ import annotations

import time
from contextlib import AsyncExitStack
from dataclasses import dataclass

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class ToolResult:
    output: str
    success: bool
    duration_ms: float


class ToolRegistry:
    """Manages MCP tool server connections and tool invocation."""

    def __init__(self):
        self._servers: dict[str, dict] = {}  # name -> {session, tools, exit_stack}
        self._tool_map: dict[str, str] = {}  # tool_name -> server_name

    async def register_server(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict | None = None,
    ):
        """Connect to an MCP server and register its tools."""
        exit_stack = AsyncExitStack()
        server_params = StdioServerParameters(command=command, args=args, env=env)
        read_stream, write_stream = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        tools_response = await session.list_tools()
        self._servers[name] = {
            "session": session,
            "tools": {t.name: t for t in tools_response.tools},
            "exit_stack": exit_stack,
        }
        for tool in tools_response.tools:
            self._tool_map[tool.name] = name
        logger.info(f"Registered MCP server '{name}' with {len(tools_response.tools)} tools")

    async def invoke(self, tool_name: str, arguments: dict) -> ToolResult:
        """Invoke a tool by name."""
        server_name = self._tool_map.get(tool_name)
        if not server_name:
            return ToolResult(f"Unknown tool: {tool_name}", False, 0)
        session = self._servers[server_name]["session"]
        start = time.monotonic()
        try:
            result = await session.call_tool(tool_name, arguments)
            duration = (time.monotonic() - start) * 1000
            output = "\n".join(c.text for c in result.content if hasattr(c, "text"))
            return ToolResult(output, True, duration)
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return ToolResult(str(e), False, duration)

    def list_tools(self) -> list[dict]:
        """List all available tools with their schemas."""
        tools = []
        for server_name, server in self._servers.items():
            for tool_name, tool in server["tools"].items():
                tools.append({
                    "name": tool_name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                    "server": server_name,
                })
        return tools

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tool_map

    async def shutdown(self):
        """Clean up all MCP server connections."""
        for name, server in self._servers.items():
            try:
                await server["exit_stack"].aclose()
                logger.debug(f"Shut down MCP server '{name}'")
            except Exception as e:
                logger.warning(f"Error shutting down server '{name}': {e}")
        self._servers.clear()
        self._tool_map.clear()
