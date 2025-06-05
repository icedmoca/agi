import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.memory import Memory


def watchdog_loop(memory: Memory, failure_threshold: int = 3, interval: int = 10) -> None:
    """Monitor memory for recurring failures and queue reflective tasks."""
    while True:
        failures = [e for e in memory.entries[-20:] if e.score <= 0]
        if len(failures) >= failure_threshold:
            goal = "Reflect on recurring failures and improve weakest component"
            task = {
                "id": f"auto_{int(datetime.now().timestamp())}",
                "goal": goal,
                "status": "pending",
                "metadata": {"tags": ["auto", "watchdog"], "created": datetime.now().isoformat()},
            }
            tasks_file = Path("tasks.jsonl")
            with tasks_file.open("a") as f:
                f.write(json.dumps(task) + "\n")
            failures.clear()
        time.sleep(interval)
