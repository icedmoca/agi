from core.memory import Memory
from core.reward import score_result

def test_append_autoscore(tmp_path):
    mem_file = tmp_path / "mem.jsonl"
    mem = Memory(file_path=str(mem_file))
    mem.append("goal", "[SUCCESS] done")
    assert mem.entries[-1].score == score_result("[SUCCESS] done")


def test_prune(tmp_path):
    mem_file = tmp_path / "mem.jsonl"
    mem = Memory(file_path=str(mem_file))
    mem.append("bad", "[ERROR]", score=-6)
    # bad entry should be pruned automatically
    assert all(e.goal != "bad" for e in mem.entries)
    mem.append("ok", "fine", score=2)
    removed = mem.prune(threshold=-5)
    # nothing else to prune
    assert removed == []

