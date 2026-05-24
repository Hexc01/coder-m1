"""LangGraph StateGraph construction and compilation."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.coder import CoderAgent
from src.agents.planner import PlannerAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.tool_agent import ToolAgent
from src.llm.client import ClaudeClient
from src.memory.long_term import LongTermMemory
from src.memory.reflection import MemoryReflector
from src.memory.retrieval import MemoryRetriever
from src.memory.short_term import ShortTermMemory
from src.state import GraphState
from src.tools.registry import ToolRegistry
from src.workflow.edges import (
    route_after_failure,
    route_after_planner,
    route_after_review,
    route_after_subtask,
    route_after_tool,
)
from src.workflow.nodes import (
    coder_node,
    handle_failure_node,
    memory_reflect_node,
    memory_retrieve_node,
    next_subtask_node,
    planner_node,
    reviewer_node,
    tool_agent_node,
    init_singletons,
)


def build_graph(
    llm_client: ClaudeClient,
    stm: ShortTermMemory,
    ltm: LongTermMemory,
    tool_registry: ToolRegistry,
) -> StateGraph:
    """Construct the multi-agent workflow graph."""

    retriever = MemoryRetriever(ltm)
    reflector = MemoryReflector(llm=llm_client, ltm=ltm)

    planner = PlannerAgent(
        name="planner", llm_client=llm_client, short_term_memory=stm, memory_retriever=retriever
    )
    coder = CoderAgent(
        name="coder", llm_client=llm_client, short_term_memory=stm, memory_retriever=retriever
    )
    reviewer = ReviewerAgent(
        name="reviewer", llm_client=llm_client, short_term_memory=stm, memory_retriever=retriever
    )
    tool_agent = ToolAgent(
        name="tool_agent", llm_client=llm_client, short_term_memory=stm,
        memory_retriever=retriever, tool_registry=tool_registry,
    )

    init_singletons(llm_client, stm, ltm, retriever, reflector, planner, coder, reviewer, tool_agent)

    graph = StateGraph(GraphState)

    graph.add_node("memory_retrieve", memory_retrieve_node)
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("tool_agent", tool_agent_node)
    graph.add_node("next_subtask", next_subtask_node)
    graph.add_node("handle_failure", handle_failure_node)
    graph.add_node("memory_reflect", memory_reflect_node)

    graph.add_edge(START, "memory_retrieve")
    graph.add_edge("memory_retrieve", "planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"coder": "coder", "tool_agent": "tool_agent", "end": END},
    )

    graph.add_edge("coder", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"next_subtask": "next_subtask", "coder": "coder", "handle_failure": "handle_failure"},
    )

    graph.add_conditional_edges(
        "next_subtask",
        route_after_subtask,
        {"coder": "coder", "tool_agent": "tool_agent", "memory_reflect": "memory_reflect"},
    )

    graph.add_conditional_edges(
        "tool_agent",
        route_after_tool,
        {"coder": "coder", "reviewer": "reviewer", "planner": "planner", "tool_agent": "tool_agent"},
    )

    graph.add_conditional_edges(
        "handle_failure",
        route_after_failure,
        {"memory_reflect": "memory_reflect"},
    )

    graph.add_edge("memory_reflect", END)

    return graph


def compile_graph(
    llm_client: ClaudeClient,
    stm: ShortTermMemory,
    ltm: LongTermMemory,
    tool_registry: ToolRegistry,
    checkpointer=None,
):
    """Compile the graph with optional checkpoint support."""
    graph = build_graph(llm_client, stm, ltm, tool_registry)
    return graph.compile(checkpointer=checkpointer)
