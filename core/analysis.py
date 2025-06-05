import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .memory import Memory


def failure_stats(memory: Memory, top_n: int = 5) -> Dict[str, List[Tuple[str, int]]]:
    """Return most common tags and files from failed memory entries."""
    failed = [e for e in memory.entries if getattr(e, "score", 0) <= 0]
    tag_counter: Counter[str] = Counter()
    file_counter: Counter[str] = Counter()
    for entry in failed:
        md = entry.metadata or {}
        tag_counter.update(md.get("tags", []))
        file_counter.update(md.get("target_files", []))
    return {
        "tags": tag_counter.most_common(top_n),
        "files": file_counter.most_common(top_n),
    }


def cluster_by_tag(memory: Memory) -> Dict[str, Dict[str, float]]:
    """Cluster memory scores by tag and return summary stats."""
    clusters: defaultdict[str, List[int]] = defaultdict(list)
    for entry in memory.entries:
        for tag in entry.metadata.get("tags", []):
            clusters[tag].append(entry.score)
    summary = {}
    for tag, scores in clusters.items():
        if scores:
            summary[tag] = {
                "count": len(scores),
                "avg_score": sum(scores) / len(scores),
            }
    return summary


def build_memory_map(memory: Memory, out_file: str = "output/memory_map.json") -> Dict[str, Dict[str, float]]:
    """Write cluster summary to JSON for visualisation."""
    summary = cluster_by_tag(memory)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    return summary
