"""core.identity – runtime self-description utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from core.memory import Memory
from core.config import MAX_RETRIES

AGENT_NAME = "Autonomo"
MODEL_NAME = "mistral-hacker"


def who_am_i(memory: Memory | None = None) -> Dict[str, Any]:
    mem = memory or Memory.latest() or Memory()
    goals_completed = len(mem.entries)
    last5_types = [e.metadata.get("type", "unknown") for e in mem.entries[-5:]] if mem.entries else []

    bio_lines = []
    reflections = Path("output/audit.log").read_text().splitlines()[-20:] if Path("output/audit.log").exists() else []
    self_desc = "Recent reflections:\n" + "\n".join(reflections[-5:])

    return {
        "agent": AGENT_NAME,
        "model": MODEL_NAME,
        "goals_completed": goals_completed,
        "last_goal_types": last5_types,
        "description": self_desc,
        "timestamp": datetime.now().isoformat(),
    }


def describe_self(memory: Memory | None = None) -> str:
    info = who_am_i(memory)
    md = (
        f"### {info['agent']}\n"
        f"*Model*: `{info['model']}`\n\n"
        f"*Goals completed*: **{info['goals_completed']}**\n\n"
        f"*Recent goal types*: {', '.join(info['last_goal_types'])}\n\n"
        f"---\n{info['description']}\n"
    )
    return md 