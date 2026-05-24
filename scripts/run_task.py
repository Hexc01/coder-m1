"""CLI entry point: run a task end-to-end through the multi-agent system."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from loguru import logger
from rich.console import Console

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.engineering.async_runner import AsyncWorkflowRunner, EventType
from src.llm.client import ClaudeClient
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from src.state import TaskStatus, create_initial_state
from src.tools.registry import ToolRegistry
from src.workflow.graph import compile_graph

console = Console()


@click.command()
@click.option("--task", required=True, help="Path to task JSON file")
@click.option("--repo", default=None, help="Target repository path")
@click.option("--thread-id", default=None, help="Thread ID for checkpoint recovery")
@click.option("--stream/--no-stream", default=True, help="Stream events or run sync")
def main(task: str, repo: str | None, thread_id: str | None, stream: bool):
    """Run a task through the multi-agent system."""
    asyncio.run(_run(task, repo, thread_id, stream))


async def _run(task_path: str, repo: str | None, thread_id: str | None, stream: bool):
    # Load task
    task_data = json.loads(Path(task_path).read_text())
    request = task_data.get("request", task_data.get("description", ""))
    task_id = task_data.get("id", "task-001")

    console.print(f"[bold green]Task:[/] {request[:100]}...")
    console.print(f"[bold blue]ID:[/] {task_id}")

    # Initialize components
    llm_client = ClaudeClient()
    stm = ShortTermMemory()
    ltm = LongTermMemory()
    tool_registry = ToolRegistry()

    # Compile graph
    app = compile_graph(llm_client, stm, ltm, tool_registry)

    # Create initial state
    initial_state = create_initial_state(
        task_id=task_id,
        request=request,
        repo_path=repo,
        max_iterations=settings.max_iterations,
    )

    # Run
    runner = AsyncWorkflowRunner(app)

    if stream:
        async for event in runner.run(initial_state, thread_id=thread_id):
            if event.event_type == EventType.NODE_START:
                console.print(f"  [cyan]-> {event.node_name}[/]")
            elif event.event_type == EventType.NODE_COMPLETE:
                console.print(f"  [green]<- {event.node_name}[/]")
            elif event.event_type == EventType.ERROR:
                console.print(f"  [red]ERROR: {event.data}[/]")
            elif event.event_type == EventType.TASK_COMPLETE:
                console.print("[bold green]Task complete![/]")
    else:
        result = await runner.run_sync(initial_state, thread_id=thread_id)
        status = result.get("task_status", "unknown")
        console.print(f"[bold]Status: {status}[/]")
        if status == TaskStatus.COMPLETED:
            console.print(f"[green]Patches: {len(result.get('patches', []))}[/]")
        else:
            console.print(f"[red]Errors: {result.get('error_log', [])}[/]")

    # Cleanup
    await tool_registry.shutdown()
    console.print("[dim]Done.[/]")


if __name__ == "__main__":
    main()
