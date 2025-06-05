"""
Retry-utility helpers shared across Agent components.
"""

from datetime import datetime
from typing import Dict, Any, List

from core.models import Task


def ensure_retry_lineage(task: Task) -> Task:
    """
    Populate the bookkeeping fields that the Agent relies on
    (`retry_lineage`, `attempt`, `previous_adjustments`).

    The function is idempotent – calling it multiple times is safe.
    """
    md: Dict[str, Any] = task.metadata

    # Chain of IDs this task descends from
    md.setdefault("retry_lineage", [])                       # type: List[str]

    # How many times this goal has been attempted
    md.setdefault("attempt", 1)

    # History of past tweaks to the goal
    md.setdefault("previous_adjustments", [])                # type: List[Dict[str, Any]]

    # Timestamp housekeeping (useful for audit/debugging)
    md.setdefault("initialized_at", datetime.now().isoformat())

    return task


__all__ = ["ensure_retry_lineage"] 