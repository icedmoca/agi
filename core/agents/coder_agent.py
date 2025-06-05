"""CoderAgent – responsible for evolving code, scoring diffs, and
committing changes.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple

from core.agents.base_agent import BaseAgent
from core.evolver import Evolver
from core.reward import score_evolution as _score_evolution
from core.utils import commit_changes, push_changes
from core.models import Task


class CoderAgent(BaseAgent):
    def __init__(self):
        super().__init__("CoderAgent", "Applies code updates")
        self.evolver = Evolver()

    # ------------------------------------------------------------------ #
    def run(self, task: Task) -> Tuple[str, str]:
        """Execute evolution for *task* and commit if successful."""
        try:
            target_file = Path(task.metadata.get("target_files", [""])[0]) if task.metadata.get("target_files") else None
            if not target_file or not target_file.exists():
                return "failed", "Target file not found"

            result_dict = self.evolver.evolve_file(filepath=str(target_file), goal=task.goal, memory=self.memory)
            status = result_dict.get("status", "failed")
            output = result_dict.get("output", "")

            diff_size = len(output)
            score = _score_evolution(task.goal, output, diff_size=diff_size, tests_passed="tests passed" in output.lower())
            task.metadata["score"] = score

            if status == "completed":
                commit_changes(f"Auto-evolution: {task.goal[:60]}")
                push_changes()

            self.log_action(task.id, "evolution", output)
            return status, output
        except Exception as e:
            self.log_action(task.id, "evolution_error", str(e))
            return "failed", str(e) 