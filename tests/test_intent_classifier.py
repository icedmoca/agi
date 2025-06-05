import types
import sys
sys.modules.setdefault('faiss', types.SimpleNamespace(IndexFlatL2=object()))
sys.modules.setdefault('ollama', types.SimpleNamespace())
sys.modules.setdefault('yaml', types.SimpleNamespace())
sys.modules.setdefault('numpy', types.SimpleNamespace())
torch_stub = types.ModuleType('torch')
torch_stub.nn = types.ModuleType('nn')
torch_stub.nn.functional = types.ModuleType('functional')
sys.modules.setdefault('torch', torch_stub)
sys.modules.setdefault('torch.nn', torch_stub.nn)
sys.modules.setdefault('torch.nn.functional', torch_stub.nn.functional)

from core.intent_classifier import classify_intent


def test_classify_intent_command(monkeypatch):
    monkeypatch.setattr('core.intent_classifier.chat_with_llm',
                        lambda **kw: '{"type":"command","value":"ls"}')
    res = classify_intent("ls")
    assert res["type"] == "command"
    assert res["value"] == "ls"


def test_classify_intent_fallback(monkeypatch):
    def boom(**kw):
        raise RuntimeError('fail')
    monkeypatch.setattr('core.intent_classifier.chat_with_llm', boom)
    res = classify_intent("ls")
    assert res["type"] in {"command", "other"}
