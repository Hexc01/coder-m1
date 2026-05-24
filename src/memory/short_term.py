from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.config import settings


@dataclass
class MemoryEntry:
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = settings.short_term_ttl
    metadata: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl_seconds


class ShortTermMemory:
    """In-memory KV store with TTL expiration for current task context."""

    def __init__(self, max_entries: int | None = None):
        self._store: dict[str, MemoryEntry] = {}
        self._max_entries = max_entries or settings.short_term_max_entries

    def put(self, key: str, value: Any, ttl: float | None = None, metadata: dict | None = None):
        """Store a value with optional TTL."""
        self._store[key] = MemoryEntry(
            key=key,
            value=value,
            ttl_seconds=ttl or settings.short_term_ttl,
            metadata=metadata or {},
        )
        self._evict_if_needed()

    def get(self, key: str) -> Any | None:
        """Retrieve a value, returning None if expired or missing."""
        entry = self._store.get(key)
        if entry is None or entry.is_expired:
            self._store.pop(key, None)
            return None
        return entry.value

    def get_all_prefix(self, prefix: str) -> dict[str, Any]:
        """Get all entries matching a key prefix."""
        return {
            k: v.value for k, v in self._store.items()
            if k.startswith(prefix) and not v.is_expired
        }

    def delete(self, key: str) -> bool:
        """Delete an entry. Returns True if it existed."""
        return self._store.pop(key, None) is not None

    def clear(self):
        """Clear all entries."""
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    def _evict_if_needed(self):
        """Remove expired entries and enforce max size."""
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]
        while len(self._store) > self._max_entries:
            oldest = min(self._store, key=lambda k: self._store[k].timestamp)
            del self._store[oldest]
