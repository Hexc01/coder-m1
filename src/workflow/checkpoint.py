"""Checkpoint/restore for task recovery using SQLite."""

from __future__ import annotations

from loguru import logger


async def create_checkpointer(db_path: str = "./data/checkpoints/checkpoints.db"):
    """Create an async SQLite checkpointer for task recovery."""
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        checkpointer = AsyncSqliteSaver.from_conn_string(db_path)
        logger.info(f"Checkpointer initialized: {db_path}")
        return checkpointer
    except ImportError:
        logger.warning("langgraph-checkpoint-sqlite not available, checkpointing disabled")
        return None


async def resume_from_checkpoint(checkpointer, thread_id: str) -> dict | None:
    """Resume a workflow from its last checkpoint."""
    if checkpointer is None:
        return None
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = await checkpointer.aget(config)
        if state:
            logger.info(f"Resumed from checkpoint: thread_id={thread_id}")
        return state
    except Exception as e:
        logger.warning(f"Failed to resume checkpoint: {e}")
        return None
