import json, types, sys, tempfile, os
sys.modules.setdefault('faiss', types.SimpleNamespace())
from core.reflection import reflect_self

def test_reflect_self_generates_goals(tmp_path, monkeypatch):
    mem_file = tmp_path / "memory.jsonl"
    # create dummy failing entries
    for i in range(5):
        mem_file.write_text("", append=False) if i == 0 else None
        with mem_file.open("a") as fp:
            fp.write(json.dumps({
                "goal": f"task {i}",
                "result": "error",
                "score": -1,
                "metadata": {"tags": ["retry"]}
            }) + "\n")
    pending = tmp_path / "pending.jsonl"
    reflect_self(str(mem_file), str(pending))
    assert pending.exists() and pending.read_text().strip() 