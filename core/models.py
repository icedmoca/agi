from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List

# Minimal first pass – more fields can be added gradually
@dataclass
class Task:
    id: str
    type: str
    goal: str
    status: str = "pending"
    priority: float = 1.0
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: str = ""                # human/tool readable output
    completed_at: str = ""          # ISO timestamp when finished

    # handy helper
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            id=data.get("id"),
            type=data.get("type"),
            goal=data.get("goal"),
            status=data.get("status", "pending"),
            priority=data.get("priority", 1.0),
            created=data.get("created", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
            result=data.get("result", ""),
            completed_at=data.get("completed_at", ""),
        )

    # ------------------------------------------------------------------ #
    # Pretty printing
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:          # noqa: D401
        goal_preview = (self.goal or "")[:30].replace("\n", " ")
        return f"<Task id={self.id} status={self.status} goal='{goal_preview}…'>"

    # -- tests still use task["goal"] style ---------------------------- #
    def __getitem__(self, key):  # type: ignore [override]
        """
        Allow dict-like access *and* tolerate positional indexing.

        • str key  → same as before (`task["goal"]`)
        • int key  → attribute by field-order (`task[0]` ⇒ `id`, `task[1]` ⇒ `type`, …)
        • other    → explicit TypeError so bugs are obvious.
        """
        if isinstance(key, str):
            return getattr(self, key)
        if isinstance(key, int):
            return list(self.__dict__.values())[key]
        raise TypeError(
            f"{self.__class__.__name__} keys must be str or int; got {type(key).__name__}"
        )


@dataclass
class MemoryEntry:
    goal: str
    result: str
    score: int
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    # convenience helpers – mirror Task API
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            goal=data.get("goal", ""),
            result=data.get("result", ""),
            score=data.get("score", 0),
            created=data.get("created", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )

    # ------------------------------------------------------------------ #
    # Pretty printing
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:          # noqa: D401
        goal_preview = (self.goal or "")[:30].replace("\n", " ")
        return f"<MemoryEntry score={self.score} goal='{goal_preview}…'>"

    # -- tests still use entry["goal"] etc. ---------------------------- #
    def __getitem__(self, key):  # type: ignore [override]
        if isinstance(key, str):
            return getattr(self, key)
        if isinstance(key, int):
            return list(self.__dict__.values())[key]
        raise TypeError(
            f"{self.__class__.__name__} keys must be str or int; got {type(key).__name__}"
        )


@dataclass
class EvolutionResult:
    file_path: str
    diff: str
    score: float
    notes: str = "" 