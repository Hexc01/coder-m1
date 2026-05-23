from __future__ import annotations

import hashlib
import json
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from src.config import settings


class LongTermMemory:
    """Persistent vector store for historical knowledge using ChromaDB."""

    COLLECTIONS = {
        "errors": "Historical error patterns and fixes",
        "patches": "Successful code patches",
        "repo_knowledge": "Repository structure and conventions",
        "task_history": "Completed task summaries and reflections",
    }

    def __init__(self, persist_dir: str | None = None):
        persist_dir = persist_dir or settings.memory_db_path
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collections: dict[str, Any] = {}
        for name, description in self.COLLECTIONS.items():
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"description": description, "hnsw:space": "cosine"},
            )
            logger.debug(f"ChromaDB collection '{name}' ready")

    def store(
        self,
        collection: str,
        document: str,
        metadata: dict | None = None,
        doc_id: str | None = None,
    ):
        """Store a document in the specified collection."""
        if collection not in self._collections:
            raise ValueError(f"Unknown collection: {collection}. Available: {list(self._collections)}")
        if doc_id is None:
            doc_id = hashlib.sha256(document.encode()).hexdigest()[:16]
        self._collections[collection].add(
            documents=[document],
            ids=[doc_id],
            metadatas=[metadata or {}],
        )

    def query(
        self,
        collection: str,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Query a collection and return results sorted by relevance."""
        if collection not in self._collections:
            raise ValueError(f"Unknown collection: {collection}")
        try:
            results = self._collections[collection].query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            logger.warning(f"Query failed on collection '{collection}': {e}")
            return []

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results.get("distances") else []
        return [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def store_error(self, error_type: str, error_msg: str, fix: str, context: str):
        """Store an error pattern with its fix."""
        self.store(
            "errors",
            document=f"Error: {error_type}: {error_msg}\nFix: {fix}\nContext: {context}",
            metadata={"error_type": error_type, "has_fix": bool(fix)},
        )

    def store_patch(self, file_path: str, description: str, patch_content: str):
        """Store a successful code patch."""
        self.store(
            "patches",
            document=f"File: {file_path}\nDescription: {description}\nPatch:\n{patch_content}",
            metadata={"file_path": file_path},
        )

    def store_repo_knowledge(self, repo_path: str, knowledge: str):
        """Store repository structure and conventions."""
        self.store(
            "repo_knowledge",
            document=knowledge,
            metadata={"repo_path": repo_path},
        )

    def store_reflection(self, task_summary: str, success: bool, lessons: list[str]):
        """Store a post-task reflection."""
        self.store(
            "task_history",
            document=task_summary,
            metadata={"success": success, "lessons": json.dumps(lessons)},
        )

    def count(self, collection: str) -> int:
        """Return the number of documents in a collection."""
        if collection not in self._collections:
            return 0
        return self._collections[collection].count()
