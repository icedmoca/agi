from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, ClassVar
from datetime import datetime

from dataclasses import asdict
from core.models import MemoryEntry
try:
    from core.vector_memory import Vectorizer
except Exception:
    Vectorizer = None
try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

class Memory:
    """
    Tiny local JSON-Lines store.

    New code should interact with MemoryEntry objects; but we keep a few
    backwards-compat affordances (e.g. `timestamp` field, filename kw-arg) so
    the historical tests continue to work.
    """

    # ------------------------------------------------------------------ #
    # Construction / persistence
    # ------------------------------------------------------------------ #
    def __init__(self, file_path: str = "memory.jsonl", **kwargs):
        # tests still call Memory(filename="foo")                         👇
        file_path = kwargs.pop("filename", file_path)
        self.path = Path(file_path)
        self.max_entries = kwargs.pop("max_entries", 1_000)
        self.vectorizer = Vectorizer() if Vectorizer else None

        # Remember the *latest* instance so helpers such as SelfHealer can
        # locate a Memory object even when one is not passed explicitly.
        Memory._latest = self                        # ←  store singleton

        # Load persisted entries; `_load()` returns a tuple
        #     (<entries>, <was_file_modified>)
        self.entries, changed = self._load()

        # Persist back only if we actually changed the on-disk data
        if changed:
            self._save()

    # A very small singleton helper (used by SelfHealer)
    _latest: ClassVar[Optional["Memory"]] = None

    @classmethod
    def latest(cls) -> Optional["Memory"]:
        return cls._latest

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def append(                        # noqa: D401  (docstring not needed)
        self,
        goal: str,
        result: str,
        score: int = 0,
        metadata: Dict[str, Any] | None = None,
    ) -> MemoryEntry:
        if score == 0:
            try:
                from core.reward import score_result
                score = score_result(result)
            except Exception:
                score = 0
        entry = MemoryEntry(goal=goal, result=result, score=score,
                            metadata=metadata or {})

        # legacy alias expected by tests
        data = entry.to_dict()
        data["timestamp"] = data["created"]

        with self.path.open("a") as fh:
            fh.write(json.dumps(data) + "\n")

        self.entries.append(entry)
        try:
            self.prune(threshold=-5)
        except Exception:
            pass
        return entry

    def get_recent(self, n: int = 5) -> List[MemoryEntry]:
        return self.entries[-n:]

    def get_recent_summary(self, n: int = 5) -> str:
        recent = self.entries[-n:]
        if not recent:
            return "No recent memory entries"
        return "\n".join(f"- {e.goal}: {e.result}" for e in recent)

    def prune(self, threshold: int = -2) -> List[MemoryEntry]:
        """
        Delete every entry whose `score` is at or below *threshold* and
        return the list of removed `MemoryEntry` objects so that callers
        (e.g. dashboards, retention bookkeeping) can decide what to do
        next.  The function keeps a disk-backed history automatically.
        """
        removed = [e for e in self.entries if getattr(e, "score", 0) <= threshold]
        self.entries = [e for e in self.entries if e not in removed]
        self._save()
        return removed

    def get_recent_context(self, goal: str, k: int = 3):
        """Return last k similar goals and their metadata"""
        similar = self.find_similar(goal, top_k=k)
        return [
            {
                "goal": e.get("goal"),
                "metadata": e.get("metadata", {})
            }
            for e in similar
        ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _load(self) -> tuple[List[MemoryEntry], bool]:
        if not self.path.exists():
            return [], False

        entries: List[MemoryEntry] = []
        changed = False
        with self.path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                entry = MemoryEntry.from_dict(obj)

                # ----- legacy files lack vector embeddings -------------
                emb = entry.metadata.get("embedding")

                if emb is None and self.vectorizer is not None:
                    emb = self.vectorizer.embed(entry.goal)
                    if hasattr(emb, "tolist"):
                        emb = emb.tolist()
                    entry.metadata["embedding"] = emb
                    changed = True

                entries.append(entry)
        return entries, changed

    # ------------------------------------------------------------------ #
    # Vector search passthrough
    # ------------------------------------------------------------------ #
    def find_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Wrapper around `Vectorizer.find_similar`.
        The vectorizer still works with plain dicts, so we convert on the fly.
        """
        rows = [e.to_dict() if hasattr(e, "to_dict") else asdict(e)
                for e in self.entries]
        if self.vectorizer:
            return self.vectorizer.find_similar(query, rows, top_k=top_k)
        # simple fallback search
        matches = []
        query_terms = set(query.lower().split())
        for entry in rows:
            entry_terms = set(entry["goal"].lower().split())
            score = len(query_terms & entry_terms)
            if score:
                matches.append((score, entry))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:top_k]]

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #
    def _json_safe(self, obj: Any) -> Any:
        """
        Convert numpy types / arrays into plain Python so json.dumps works.
        """
        if np is not None:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._json_safe(v) for v in obj]
        return obj

    def _save(self) -> None:
        """Persist `self.entries` to disk in JSON-serialisable form."""
        with self.path.open("w") as fh:
            for e in self.entries:
                data = self._json_safe(e.to_dict())
                data["timestamp"] = data["created"]          # legacy alias
                fh.write(json.dumps(data) + "\n")