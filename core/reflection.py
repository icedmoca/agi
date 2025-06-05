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