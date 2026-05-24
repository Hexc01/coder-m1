from __future__ import annotations

from loguru import logger

from src.config import settings
from src.memory.long_term import LongTermMemory


class MemoryRetriever:
    """RAG retrieval: query long-term memory for relevant context."""

    def __init__(self, long_term: LongTermMemory):
        self.ltm = long_term

    async def query(self, task_description: str, n_results: int | None = None) -> str:
        """Query all collections and return a formatted context string."""
        n = n_results or settings.retrieval_top_n
        results: list[tuple[float, str]] = []

        for collection in LongTermMemory.COLLECTIONS:
            try:
                hits = self.ltm.query(collection, task_description, n_results=min(n, 3))
                for hit in hits:
                    results.append((hit["distance"], f"[{collection}] {hit['document']}"))
            except Exception as e:
                logger.warning(f"Retrieval failed for collection '{collection}': {e}")

        results.sort(key=lambda x: x[0])
        context_parts = [r[1] for r in results[:n]]
        return "\n---\n".join(context_parts) if context_parts else ""

    async def find_similar_tasks(self, task_description: str) -> list[dict]:
        """Find tasks similar to the current one."""
        hits = self.ltm.query("task_history", task_description, n_results=5)
        return [
            {
                "document": h["document"],
                "metadata": h["metadata"],
                "similarity": 1 - h["distance"],
            }
            for h in hits
            if (1 - h["distance"]) > settings.similarity_threshold
        ]

    async def find_relevant_patches(self, code_description: str, n: int = 3) -> list[dict]:
        """Find patches relevant to a code change description."""
        hits = self.ltm.query("patches", code_description, n_results=n)
        return [
            {"document": h["document"], "metadata": h["metadata"], "distance": h["distance"]}
            for h in hits
        ]

    async def find_error_fixes(self, error_description: str, n: int = 3) -> list[dict]:
        """Find historical fixes for similar errors."""
        hits = self.ltm.query("errors", error_description, n_results=n)
        return [
            {"document": h["document"], "metadata": h["metadata"], "distance": h["distance"]}
            for h in hits
        ]
