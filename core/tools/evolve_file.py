from typing import Any
from core.memory import Memory
from core.models import Task

def evolve_file(
    task: Task,
    memory: Memory | None = None,   # memory can be None for unit-tests
    **kwargs,
) -> str:
    """
    Dummy/stub implementation used by the unit-tests.

    In production this function would call the real code-evolution
    pipeline; during tests we only need to guarantee that it returns
    a string containing either "[SUCCESS]" or "[ERROR]".
    """

    goal: str = task.goal
    notes: str = task.metadata.get("notes", "")

    # Extremely light-weight simulation
    if not goal:
        return "[ERROR] Empty goal received"

    diff_preview = f"# diff for {goal} (notes: {notes})\n---\n+++"

    # Always succeed in the stub so tests can match "[SUCCESS]"
    return f"[SUCCESS] Evolution simulated OK\n{diff_preview}" 