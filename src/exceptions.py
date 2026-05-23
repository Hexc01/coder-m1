class CoderM1Error(Exception):
    """Base exception for the coder-m1 system."""


class AgentError(CoderM1Error):
    """Error during agent execution."""

    def __init__(self, agent_name: str, message: str, subtask_id: str | None = None):
        self.agent_name = agent_name
        self.subtask_id = subtask_id
        super().__init__(f"[{agent_name}] {message}")


class ToolError(CoderM1Error):
    """Error during tool invocation."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}': {message}")


class LLMError(CoderM1Error):
    """Error from LLM API call."""


class MemoryError(CoderM1Error):
    """Error in memory system."""


class WorkflowError(CoderM1Error):
    """Error in workflow execution."""


class MaxIterationsError(WorkflowError):
    """Circuit breaker: workflow exceeded max iterations."""
