import json
from collections import defaultdict
from pathlib import Path


from .chat import score_llm_output, chat_with_llm, set_system_prompt, get_system_prompt


def evolve_prompt(prompt: str) -> str:
    """Use the LLM to suggest an improved system prompt."""
    try:
        user = f"Improve this system prompt for clarity and reliability:\n{prompt}"
        improved = chat_with_llm(user, "You are a prompt engineer.")
        return improved.strip() or prompt
    except Exception:
        return prompt


def scan_history_and_evolve(
    history_path: str = "output/llm_history.jsonl",
    low_score: float = 0.4,
    min_count: int = 3,
) -> None:
    """Scan LLM history and evolve the global system prompt on repeated failures."""
    path = Path(history_path)
    if not path.exists():
        return
    counts: defaultdict[str, int] = defaultdict(int)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = score_llm_output(entry.get("output", ""))
            if score < low_score:
                counts[entry.get("system_prompt", "")] += 1
    for prompt, count in counts.items():
        if count >= min_count:
            new_prompt = evolve_prompt(prompt)
            if new_prompt != prompt:
                set_system_prompt(new_prompt)
                out = Path("output/system_prompt.txt")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(new_prompt)
                break
