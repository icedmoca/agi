from pathlib import Path
import json
from typing import Dict, List
from datetime import datetime

class TaskSanitizer:
    def __init__(self, tasks_file: str = "tasks.jsonl"):
        self.tasks_file = Path(tasks_file)
        
    def sanitize_tasks(self) -> None:
        """Sanitize tasks.jsonl to use current format"""
        if not self.tasks_file.exists():
            return
            
        # Read all tasks
        tasks = []
        with self.tasks_file.open() as f:
            for line in f:
                if line.strip():
                    try:
                        task = json.loads(line)
                        tasks.append(self.sanitize_task(task))
                    except json.JSONDecodeError:
                        continue
                        
        # Write sanitized tasks back
        with self.tasks_file.open('w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')
                
    def sanitize_task(self, task: Dict) -> Dict:
        """Sanitize a single task to current format"""
        # Add version if missing
        if "version" not in task:
            task["version"] = "1.0"
            
        # Remove legacy fields
        legacy_fields = ["test_passed", "old_format", "legacy_score"]
        for field in legacy_fields:
            task.pop(field, None)
            
        # Ensure required fields
        required = {
            "id": f"task_{int(datetime.now().timestamp())}",
            "type": "evolution",
            "goal": "",
            "priority": 1,
            "status": "pending",
            "created": datetime.now().isoformat(),
            "result": None
        }
        
        for key, default in required.items():
            if key not in task:
                task[key] = default
                
        return task 