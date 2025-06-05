"""HealerAgent – monitors alerts and attempts automated resolutions."""

from __future__ import annotations

from typing import Tuple
from core.agents.base_agent import BaseAgent
from core.self_heal import SelfHealer, ResolutionStatus


class HealerAgent(BaseAgent):
    def __init__(self):
        super().__init__("HealerAgent", "Handles system self-healing")
        self.healer = SelfHealer()

    # ------------------------------------------------------------------ #
    def handle_alert(self, alert: dict) -> Tuple[str, str]:
        status, msg = self.healer.attempt_resolution(alert)
        self.log_action(None, "self_heal", f"{alert.get('type')}: {status} – {msg}")
        return status, msg 