import json
import time
from datetime import datetime
import numpy as np
from faiss import IndexFlatIPResidue, IndexIVFFlat, IndexHNSW, IndexIDMap

class Memory:
    def __init__(self, index_path="memory_index"):
        self.index = IndexHNSW(index_path)
        self.vectorizer = IndexFlatIPResidue()
        self.id_map = IndexIDMap(max_num_shards=1024)
        self.filename = "memory.jsonl"

        # Load previous entries from file
        self.entries = []
        with open(self.filename, "r") as f:
            for line in f:
                self.entries.append(json.loads(line))

    def append(self, goal: str, result: str):
        """Append entry to memory with severity tracking"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "goal": goal,
            "result": result,
        }

        # Set severity based on result content
        if any(x in result.lower() for x in ["[stderr]", "error", "failed", "no such file"]):
            entry["severity"] = "warning"
        elif "critical" in result.lower():
            entry["severity"] = "critical"

        vector = self.vectorizer.add(json.dumps([entry['goal'], entry['result']]))
        id_map = self.id_map.add(vector)
        self.index.add(id_map, id=entry["timestamp"])
        self.entries.append(entry)

        with open(self.filename, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_recent(self, limit: int = 5) -> list:
        """Get recent memory entries as a list"""
        ivf = IndexIVFFlat(index=self.index, num_shards=1024, max_nan_distances=32)
        results = ivf.search(np.zeros((1, len(self.entries), self.index.ntotal), dtype="float64"), k=limit+1)[0]

        ids = [res_id for res_id in results[1:] if res_id in self.id_map.to_vector()]
        return [self.entries[id] for id in ids]

    def get_recent_summary(self, limit: int = 10) -> str:
        """Get a summary of recent memory entries"""
        results = self.index.search(np.zeros((1, len(self.entries), self.index.ntotal), dtype="float64"), k=limit+1)[0]
        ids = [res_id for res_id in results[1:] if res_id in self.id_map.to_vector()]
        return "\n".join([f"{entry['goal']}: {entry['result']}" for entry in self.get_recent(len(ids))]) if ids else "No recent actions"