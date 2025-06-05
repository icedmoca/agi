from core.memory import Memory
from core.analysis import failure_stats, build_memory_map


def test_failure_stats_and_map(tmp_path):
    mem = Memory(file_path=str(tmp_path / "mem.jsonl"))
    mem.append("a", "fail", score=-1, metadata={"tags": ["bug"], "target_files": ["f.py"]})
    mem.append("b", "fail", score=-1, metadata={"tags": ["bug", "ui"], "target_files": ["f.py", "g.py"]})
    stats = failure_stats(mem)
    assert stats["tags"][0][0] == "bug"
    out_file = tmp_path / "map.json"
    summary = build_memory_map(mem, out_file=str(out_file))
    assert "bug" in summary
    assert out_file.exists()
