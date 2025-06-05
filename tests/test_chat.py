import sys, types
sys.modules.setdefault('ollama', types.SimpleNamespace())
sys.modules.setdefault('yaml', types.SimpleNamespace())
from core.chat import _log_llm_history
from pathlib import Path
import json

def test_log_llm_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _log_llm_history("sys", "user", "out", "model")
    data = json.loads(Path("output/llm_history.jsonl").read_text())
    assert data["model"] == "model"
