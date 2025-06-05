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
from core.evolver import safe_apply_evolution
from core.memory import Memory
from pathlib import Path

def test_evolve_fetch_url(tmp_path):
    # create temp tool file with simple fetch_url
    tool_file = tmp_path / "tool.py"
    tool_file.write_text("def fetch_url(url):\n    import requests\n    return requests.get(url).text\n")

    mem = Memory(file_path=str(tmp_path / "mem.jsonl"))
    goal = "Update fetch_url tool to handle JSON decoding errors"
    new_code = "def fetch_url(url):\n    import requests, json\n    resp = requests.get(url)\n    try:\n        return json.dumps(resp.json())\n    except Exception:\n        return resp.text\n"
    result = safe_apply_evolution(str(tool_file), new_code, goal, mem)
    assert result["status"] == "success"
    assert "json.dumps" in tool_file.read_text()
    # memory logged
    assert mem.entries
    log = Path("evolution_log.md").read_text()
    assert "fetch_url" in log
