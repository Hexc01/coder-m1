from __future__ import annotations

from abc import ABC, abstractmethod

from src.llm.client import ClaudeClient
from src.memory.retrieval import MemoryRetriever
from src.memory.short_term import ShortTermMemory
from src.state import AgentMessage, GraphState


class BaseAgent(ABC):
    """Abstract base for all agents in the system."""

    def __init__(
        self,
        name: str,
        llm_client: ClaudeClient,
        short_term_memory: ShortTermMemory,
        memory_retriever: MemoryRetriever,
    ):
        self.name = name
        self.llm = llm_client
        self.stm = short_term_memory
        self.retriever = memory_retriever

    @abstractmethod
    async def execute(self, state: GraphState) -> dict:
        """Execute this agent's logic and return state updates."""
        ...

    def _build_messages(self, state: GraphState, system_prompt: str) -> list[dict]:
        """Build the message list for the LLM call."""
        messages = []
        for msg in state["messages"][-20:]:
            role = "assistant" if msg.sender != "user" else "user"
            messages.append({"role": role, "content": msg.content})
        return messages

    def _emit_message(
        self, receiver: str, content: str, msg_type: str, subtask_id: str | None = None
    ) -> list[AgentMessage]:
        """Create an outgoing AgentMessage."""
        return [AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=msg_type,  # type: ignore[arg-type]
            subtask_id=subtask_id,
        )]
