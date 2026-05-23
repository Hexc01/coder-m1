# Coder-M1: Multi-Agent Software Engineering System

## Overview
Memory-Augmented Multi-Agent system for autonomous software engineering tasks.
LangGraph DAG workflow orchestrating 4 agent types with ChromaDB vector memory.
LLM backend: mimo-v2.5-pro via Anthropic-compatible API.

## Quick Start
```bash
pip install -e ".[dev]"
cp .env.example .env   # then add ANTHROPIC_API_KEY
python scripts/run_task.py --task examples/hello_world_task.json
pytest tests/ -v
```

## Project Structure
- `src/agents/` — Planner, Coder, Reviewer, Tool-Agent implementations
- `src/workflow/` — LangGraph StateGraph construction, nodes, edges, checkpointing
- `src/memory/` — Short-term (in-memory KV) + long-term (ChromaDB) + RAG retrieval + reflection
- `src/tools/` — MCP tool registry (shell, git, filesystem)
- `src/engineering/` — AST code indexer, git patch manager, async event runner
- `src/llm/` — Claude API client wrapper and system prompts
- `src/state.py` — GraphState TypedDict and all Pydantic data models
- `src/config.py` — Global config (pydantic-settings)
- `tests/` — pytest test suite (unit + integration)
- `scripts/` — CLI entry points
- `examples/` — Example task JSON files
- `data/` — Runtime data (checkpoints, memory DB, indexes) — gitignored

## Architecture
- **State**: `GraphState` TypedDict flows through all nodes; `Annotated[list, operator.add]` reducers for append-only fields (messages, patches, tool_calls)
- **Agents**: Each agent extends `BaseAgent`, implements `async execute(state) -> dict` returning partial state updates
- **Workflow**: LangGraph `StateGraph` with 8 nodes and 5 conditional edge functions
- **Memory**: Short-term (KV with TTL) + long-term (ChromaDB, 4 collections) + RAG retrieval + post-task reflection
- **Tools**: MCP servers via stdio; Tool Agent executes pending requests from other agents

## Key Patterns
- Agents return partial state dicts — LangGraph merges them automatically
- Tool calls are async: agent sets `pending_tool_request`, Tool Agent node executes and routes back
- Checkpoints auto-save after each node (SQLite backend) for crash recovery
- `iteration_count` circuit breaker prevents infinite loops (default max: 50)

## Configuration
All config via env vars or `src/config.py`:
- `ANTHROPIC_API_KEY` — Required (mimo API key)
- `ANTHROPIC_BASE_URL` — Default: https://token-plan-cn.xiaomimimo.com/anthropic
- `MODEL_NAME` — Default: mimo-v2.5-pro
- `MAX_ITERATIONS` — Default: 50
- `MEMORY_DB_PATH` — Default: ./data/memory_db
- `CHECKPOINT_DB_PATH` — Default: ./data/checkpoints/checkpoints.db

## Development Conventions
- Python 3.10+, type hints everywhere
- `loguru` for logging, `rich` for CLI output
- `pydantic` for all data models
- `pytest` + `pytest-asyncio` for tests (asyncio_mode = "auto")
- `ruff` for linting (line-length 100)
- Commit messages: imperative mood, e.g. "Add planner agent implementation"

## Scope Notes
- **RL optimization (Multi-Agent RL, reward function, group-based policy optimization) is deferred** — focus on core multi-agent system + memory + engineering first. Do NOT implement RL-related code until core is stable.
