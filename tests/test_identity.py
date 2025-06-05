import types, sys
sys.modules.setdefault('faiss', types.SimpleNamespace())
sys.modules.setdefault('ollama', types.SimpleNamespace())

from core.identity import who_am_i, describe_self

def test_who_am_i():
    info = who_am_i()
    assert isinstance(info, dict)
    assert 'agent' in info and 'model' in info

def test_describe_self():
    md = describe_self()
    assert md.startswith('###') 