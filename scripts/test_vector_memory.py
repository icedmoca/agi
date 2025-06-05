from unittest.mock import patch
from core.memory import Memory


def test_vector_memory_search(tmp_path):
    mem_file = tmp_path / "memory.jsonl"
    memory = Memory(filename=str(mem_file))

    memory.append(goal="Test goal", result="Test result")

    similar = memory.find_similar("Test goal", top_k=1)
    assert similar
    assert similar[0]['goal'] == "Test goal"

    with patch('builtins.input', return_value='q'):
        pass
