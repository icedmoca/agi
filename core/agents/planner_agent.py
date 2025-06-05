"""PlannerAgent – decomposes incoming high-level goals into actionable
sub-goals and tool invocations.
"""

from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime
import json

from core.agents.base_agent import BaseAgent
from core.utils import suggest_tags
from core.chat import chat_with_llm
from core.models import Task


class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("PlannerAgent", "Plans and decomposes goals")

    # ------------------------------------------------------------------ #
    def parse_goal(self, goal: str) -> List[str]:
        """Use LLM to break *goal* into ranked sub-goals."""
        prompt = (
            "Break the following product goal into 1-3 concrete developer sub-goals.\n"
            "Return each sub-goal on its own line without numbering.\n\n" + goal
        )
        try:
            raw = chat_with_llm(prompt, system_prompt="You are a senior engineering planner.")
            lines = [l.strip("- •\t ") for l in raw.splitlines() if l.strip()]
            return lines[:3]
        except Exception:
            return [goal]

    # ------------------------------------------------------------------ #
    def create_tasks(self, goal: str, origin: str = "user") -> List[Task]:
        sub_goals = self.parse_goal(goal)
        tasks: List[Task] = []
        for sg in sub_goals:
            task = Task(
                id=f"task_{int(datetime.now().timestamp())}",
                type="evolution" if any(k in sg.lower() for k in ("code", "file", "evolve", "refactor")) else "chat",
                goal=sg,
                priority=1.0,
                metadata={
                    "tags": suggest_tags(sg),
                    "origin": origin,
                },
            )
            tasks.append(task)
            self.log_action(task.id, "generated_subgoal", sg)
        return tasks 