"""Node functions for the LangGraph workflow."""

from __future__ import annotations

from loguru import logger

from src.memory.long_term import LongTermMemory
from src.memory.reflection import MemoryReflector
from src.memory.retrieval import MemoryRetriever
from src.memory.short_term import ShortTermMemory
from src.state import GraphState, TaskStatus


# --- Singleton holders (initialized by graph.py) ---
_llm_client = None
_stm: ShortTermMemory | None = None
_ltm: LongTermMemory | None = None
_retriever: MemoryRetriever | None = None
_reflector: MemoryReflector | None = None
_planner = None
_coder = None
_reviewer = None
_tool_agent = None


def init_singletons(
    llm_client,
    stm: ShortTermMemory,
    ltm: LongTermMemory,
    retriever: MemoryRetriever,
    reflector: MemoryReflector,
    planner,
    coder,
    reviewer,
    tool_agent,
):
    global _llm_client, _stm, _ltm, _retriever, _reflector
    global _planner, _coder, _reviewer, _tool_agent
    _llm_client = llm_client
    _stm = stm
    _ltm = ltm
    _retriever = retriever
    _reflector = reflector
    _planner = planner
    _coder = coder
    _reviewer = reviewer
    _tool_agent = tool_agent


async def memory_retrieve_node(state: GraphState) -> dict:
    """Retrieve relevant context from long-term memory before planning."""
    logger.info("[Node] memory_retrieve")
    context = await _retriever.query(state["original_request"], n_results=5)
    similar = await _retriever.find_similar_tasks(state["original_request"])
    return {
        "memory_context": context,
        "similar_past_tasks": similar,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


async def planner_node(state: GraphState) -> dict:
    """Planner agent: decompose task into subtasks."""
    logger.info("[Node] planner")
    return await _planner.execute(state)


async def coder_node(state: GraphState) -> dict:
    """Coder agent: generate code patch for current subtask."""
    logger.info("[Node] coder")
    return await _coder.execute(state)


async def reviewer_node(state: GraphState) -> dict:
    """Reviewer agent: review current patch."""
    logger.info("[Node] reviewer")
    return await _reviewer.execute(state)


async def tool_agent_node(state: GraphState) -> dict:
    """Tool agent: execute pending tool request."""
    logger.info("[Node] tool_agent")
    return await _tool_agent.execute(state)


async def next_subtask_node(state: GraphState) -> dict:
    """Advance to the next subtask or complete the task."""
    logger.info("[Node] next_subtask")
    subtasks = list(state["subtasks"])
    idx = state["current_subtask_index"]
    subtasks[idx].status = TaskStatus.COMPLETED
    next_idx = idx + 1
    if next_idx >= len(subtasks):
        return {
            "subtasks": subtasks,
            "task_status": TaskStatus.COMPLETED,
            "current_subtask_index": next_idx,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    return {
        "subtasks": subtasks,
        "current_subtask_index": next_idx,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


async def handle_failure_node(state: GraphState) -> dict:
    """Handle subtask failure after max retries."""
    logger.info("[Node] handle_failure")
    subtasks = list(state["subtasks"])
    idx = state["current_subtask_index"]
    subtasks[idx].status = TaskStatus.FAILED
    return {
        "subtasks": subtasks,
        "task_status": TaskStatus.FAILED,
        "error_log": [f"Subtask {subtasks[idx].id} failed after max retries"],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


async def memory_reflect_node(state: GraphState) -> dict:
    """Post-task reflection: summarize what worked/failed."""
    logger.info("[Node] memory_reflect")
    try:
        await _reflector.reflect(state)
    except Exception as e:
        logger.error(f"Reflection failed: {e}")
    return {"iteration_count": state.get("iteration_count", 0) + 1}
