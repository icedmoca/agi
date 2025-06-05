from core.vector_memory import VectorMemory
from datetime import datetime, timedelta
import os
import json
from typing import List, Dict

class ReflectionAgent:
    def __init__(self, memory: VectorMemory, log_files=None):
        self.memory = memory
        self.log_files = log_files or {
            "changes": "output/file_changes.log",
            "agent": "output/agent_loop.log",
            "errors": "output/hash_check_errors.log"
        }
        
    def get_recent_logs(self, file_path: str, window: int = 20) -> List[str]:
        """Get recent log entries from a file"""
        if not os.path.exists(file_path):
            return []
            
        with open(file_path, "r") as f:
            lines = f.readlines()[-window:]
            return [line.strip() for line in lines if line.strip()]
            
    def analyze_patterns(self, logs: List[str]) -> Dict:
        """Analyze logs for patterns and anomalies"""
        patterns = {
            "errors": 0,
            "changes": 0,
            "successes": 0,
            "anomalies": []
        }
        
        for log in logs:
            if "ERROR" in log or "failed" in log.lower():
                patterns["errors"] += 1
            elif "MODIFIED" in log or "CREATED" in log:
                patterns["changes"] += 1
            elif "✅" in log or "success" in log.lower():
                patterns["successes"] += 1
                
            # Detect potential anomalies
            if "unexpected" in log.lower() or "warning" in log.lower():
                patterns["anomalies"].append(log)
                
        return patterns
        
    def reflect(self, window: int = 20) -> str:
        """Generate reflection summary of recent activity"""
        summary = []
        all_logs = []
        
        # Collect logs from all sources
        for log_type, file_path in self.log_files.items():
            logs = self.get_recent_logs(file_path, window)
            if logs:
                all_logs.extend(logs)
                # Add to vector memory for future reference
                for log in logs:
                    self.memory.add(f"[{log_type}] {log}")
                    
        # Analyze patterns
        patterns = self.analyze_patterns(all_logs)
        
        # Generate summary
        summary.append("🪞 System Self-Reflection")
        summary.append("=" * 40)
        
        if patterns["errors"] > 0:
            summary.append(f"⚠️ Detected {patterns['errors']} errors/failures")
        if patterns["changes"] > 0:
            summary.append(f"📝 Observed {patterns['changes']} file changes")
        if patterns["successes"] > 0:
            summary.append(f"✅ Recorded {patterns['successes']} successful operations")
            
        if patterns["anomalies"]:
            summary.append("\n🔍 Potential Anomalies:")
            for anomaly in patterns["anomalies"][:3]:  # Show top 3
                summary.append(f"  • {anomaly}")
                
        # Search for related past events
        if all_logs:
            recent_event = all_logs[-1]
            related = self.memory.search(recent_event, top_k=2)
            if len(related) > 1:  # If we found similar past events
                summary.append("\n📚 Related Past Events:")
                for event in related[1:]:  # Skip the current event
                    summary.append(f"  • {event}")
                    
        return "\n".join(summary) 

# --------------------------------------------------------------------------- #
# New public helper for autonomous loop
# --------------------------------------------------------------------------- #


def reflect_self(memory_file: str = "memory.jsonl", pending: str = "pending_goals.jsonl") -> None:
    """Scan *memory_file* for repeated failures / low scores and append new
    meta-goals into *pending_goals.jsonl* when patterns are detected."""

    from collections import Counter
    from pathlib import Path

    mem_path = Path(memory_file)
    if not mem_path.exists():
        return

    entries = [json.loads(l) for l in mem_path.read_text().splitlines() if l.strip()]

    # Gather tags and scores
    fail_tags: Counter[str] = Counter()
    recent_scores = []
    unused_tools: Counter[str] = Counter()

    for e in entries[-200:]:  # last 200 entries
        score = e.get("score", 0)
        recent_scores.append(score)
        if score < 0:
            for t in e.get("metadata", {}).get("tags", []):
                fail_tags[t] += 1

        if e.get("goal", "").startswith("tool:"):
            tool_name = e["goal"].split(":", 1)[1]
            unused_tools[tool_name] += 1

    # simple heuristics
    new_goals = []
    for tag, cnt in fail_tags.items():
        if cnt >= 3:
            new_goals.append(f"Improve handling of tasks related to '{tag}' to reduce repeated failures")

    if len(recent_scores) >= 20 and sum(1 for s in recent_scores[-20:] if s > 0) < 5:
        new_goals.append("Investigate declining success rate and enhance retry logic")

    # If tools unused recently suggest exploration
    for tool, cnt in unused_tools.items():
        if cnt < 2:
            new_goals.append(f"Create tasks that utilise the '{tool}' tool to validate its utility")

    if not new_goals:
        return

    pending_path = Path(pending)
    pending_path.touch(exist_ok=True)
    with pending_path.open("a") as fp:
        for g in new_goals:
            obj = {
                "goal": g,
                "origin": "reflection",
                "created": datetime.now().isoformat(),
            }
            fp.write(json.dumps(obj) + "\n")
    # also log
    print(f"[Reflection] Injected {len(new_goals)} goal(s)") 