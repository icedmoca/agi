import json
import types
import sys

sys.modules.setdefault('ollama', types.SimpleNamespace())

from core import prompt_manager


def test_scan_history_and_evolve(tmp_path, monkeypatch):
    history = tmp_path / "llm_history.jsonl"
    entries = [{"system_prompt": "old", "user_input": "u", "output": "bad"}] * 3
    history.write_text("\n".join(json.dumps(e) for e in entries))

    called = {}
    monkeypatch.setattr(prompt_manager, "chat_with_llm", lambda sp, ui: "new")
    monkeypatch.setattr(prompt_manager, "set_system_prompt", lambda p: called.setdefault("prompt", p))

    prompt_manager.scan_history_and_evolve(str(history), low_score=0.6, min_count=3)
    assert called.get("prompt") == "new"
