import sys, types
sys.modules.setdefault('faiss', types.SimpleNamespace())
sys.modules.setdefault('ollama', types.SimpleNamespace())
torch_stub = types.ModuleType('torch')
torch_stub.nn = types.ModuleType('nn')
torch_stub.nn.functional = types.ModuleType('functional')
sys.modules.setdefault('torch', torch_stub)
sys.modules.setdefault('torch.nn', torch_stub.nn)
sys.modules.setdefault('torch.nn.functional', torch_stub.nn.functional)
sys.modules.setdefault('yaml', types.SimpleNamespace())
from core.evolver import sanitize_llm_output, safe_apply_evolution
from core.memory import Memory
import tempfile
from pathlib import Path
import ast


def test_sanitize_llm_output():
    raw = "```python\nprint('hi')\n```"
    cleaned = sanitize_llm_output(raw)
    assert cleaned.strip() == "print('hi')"
    ast.parse(cleaned)  # should be valid Python


def test_safe_apply_evolution(tmp_path):
    mem = Memory(file_path=str(tmp_path / "mem.jsonl"))
    file_path = tmp_path / "code.py"
    file_path.write_text("print('old')\n")
    result = safe_apply_evolution(str(file_path), "print('new')", "update", mem)
    assert result["status"] == "success"
    assert "print('new')" in file_path.read_text()
    log = Path("evolution_log.md").read_text()
    assert "code.py" in log


def test_safe_apply_evolution_create(tmp_path):
    mem = Memory(file_path=str(tmp_path / "mem.jsonl"))
    file_path = tmp_path / "new.py"
    result = safe_apply_evolution(str(file_path), "print('hi')", "create", mem)
    assert result["status"] == "success"
    assert file_path.exists()
