"""core.agents.base_agent

Common base-class for all specialised agents (Planner, Coder, Healer, …).
Keeps references to *Memory* and offers a simple *log_action* helper so
all agents write to a shared activity log.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from core.memory import Memory


class BaseAgent:
    """Light-weight common functionality shared by sub-agents."""

    activity_log = Path("output/agent_activity.jsonl")

    def __init__(self, name: str, role: str):
        self.name   = name
        self.role   = role
        self.memory = Memory.latest() or Memory()

    # ------------------------------------------------------------------ #
    # Logging helpers                                                    #
    # ------------------------------------------------------------------ #
    def log_action(self, task_id: str | None, action: str, result: str) -> None:
        entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "role": self.role,
            "task_id": task_id,
            "action": action,
            "result": result[:500],  # truncate to keep file small
        }
        self.activity_log.parent.mkdir(parents=True, exist_ok=True)
        with self.activity_log.open("a") as fp:
            fp.write(json.dumps(entry) + "\n") 