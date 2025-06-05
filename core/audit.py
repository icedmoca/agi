import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

class AuditLogger:
    def __init__(self, audit_file: str = "output/audit.log"):
        self.audit_file = Path(audit_file)
        self.audit_file.parent.mkdir(exist_ok=True)
        
    def log_event(self, 
                  event_type: str,
                  goal: Optional[str] = None,
                  result: Optional[str] = None,
                  score: Optional[int] = None,
                  attempt: Optional[int] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a structured audit event"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "goal": goal,
            "result": result
        }
        
        # Add optional fields if present
        if score is not None:
            entry["score"] = score
        if attempt is not None:
            entry["attempt"] = attempt
        if metadata:
            entry.update(metadata)
            
        # Write entry
        with self.audit_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")
            
    def log_summary(self, window_hours: Optional[float] = None) -> Dict[str, Any]:
        """Parse audit.log and return summary statistics"""
        if not self.audit_file.exists():
            return {}

        total = 0
        success = 0
        failed = 0
        retried = 0
        scores = []
        unique_goals = set()
        
        cutoff = None
        if window_hours:
            cutoff = datetime.now().timestamp() - (window_hours * 3600)

        with self.audit_file.open() as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    
                    # Apply time window filter if specified
                    if cutoff:
                        entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                        if entry_time < cutoff:
                            continue
                    
                    total += 1
                    if entry.get("goal"):
                        unique_goals.add(entry["goal"])

                    if entry.get("event") == "task_result":
                        if "[SUCCESS]" in str(entry.get("result", "")):
                            success += 1
                        else:
                            failed += 1

                        if "score" in entry:
                            scores.append(entry["score"])

                    elif entry.get("event") == "retry_queued":
                        retried += 1

                except Exception:
                    continue

        return {
            "total_tasks": total,
            "unique_goals": len(unique_goals),
            "successes": success,
            "failures": failed,
            "retries": retried,
            "success_rate": round(success / total * 100, 1) if total > 0 else 0,
            "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
            "total_score": sum(scores) if scores else 0
        } 