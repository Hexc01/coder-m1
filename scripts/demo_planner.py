"""Demo: run the Planner agent standalone."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from src.agents.planner import PlannerAgent
from src.config import settings
from src.llm.client import ClaudeClient
from src.memory.long_term import LongTermMemory
from src.memory.retrieval import MemoryRetriever
from src.memory.short_term import ShortTermMemory
from src.state import TaskStatus, create_initial_state

console = Console()


async def demo():
    llm = ClaudeClient()
    stm = ShortTermMemory()
    ltm = LongTermMemory()
    retriever = MemoryRetriever(ltm)

    planner = PlannerAgent(
        name="planner", llm_client=llm, short_term_memory=stm, memory_retriever=retriever
    )

    request = "Create a Python function that calculates the Fibonacci sequence up to n terms"
    state = create_initial_state(task_id="demo-001", request=request)

    console.print(f"[bold]Request:[/] {request}\n")
    console.print("[cyan]Running Planner...[/]")

    result = await planner.execute(state)

    console.print(f"\n[green]Created {len(result.get('subtasks', []))} subtasks:[/]")
    for st in result.get("subtasks", []):
        console.print(f"  [{st.id}] {st.description} (agent: {st.assigned_agent})")


if __name__ == "__main__":
    asyncio.run(demo())
