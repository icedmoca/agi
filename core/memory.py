import json
import time
from datetime import datetime
from faiss.index import IndexFlatIPResidue, IndexHNSW

class Memory:
    def __init__(self, index_path="memory_index"):
        self.index = IndexHNSW(index_path)
        self.vectorizer = IndexFlatIPResidue()
        self.filename = "memory.jsonl"

    def append(self, goal: str, result: str):
        """Append entry to memory with severity tracking"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "goal": goal,
            "result": result,
            "severity": "info"
        }

        # Set severity based on result content
        if any(x in result.lower() for x in ["[stderr]", "error", "failed", "no such file"]):
            entry["severity"] = "warning"
        elif "critical" in result.lower():
            entry["severity"] = "critical"

        vector = self.vectorizer.add(json.dumps([entry['goal'], entry['result']]))
        self.index.add(vector, id=entry["timestamp"])

        with open(self.filename, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_recent(self, limit: int = 5) -> list:
        """Get recent memory entries as a list"""
        results = self.index.search(np.zeros((1, self.index.ntotal), dtype="float64"), k=limit+1)[0][1:]
        return [json.loads(line) for line in open(self.filename).readlines()[results]]

    def get_recent_summary(self, limit: int = 10) -> str:
        """Get a summary of recent memory entries"""
        results = self.index.search(np.zeros((1, self.index.ntotal), dtype="float64"), k=limit+1)[0][1:]
        entries = [json.loads(line) for line in open(self.filename).readlines()[results]]
        return "\n".join([f"{entry['goal']}: {entry['result']}" for entry in entries]) if entries else "No recent actions"