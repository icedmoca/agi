import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

def inject_goal(goal: str, goal_type: str = "evolution", priority: int = 1):
    """Inject a new goal into tasks.jsonl"""
    task = {
        "id": f"task_{int(datetime.now().timestamp())}",
        "type": goal_type,
        "goal": goal,
        "priority": priority,
        "status": "pending",
        "created": datetime.now().isoformat(),
        "result": None
    }
    
    tasks_file = Path("tasks.jsonl")
    
    # Create file if doesn't exist
    if not tasks_file.exists():
        tasks_file.write_text("")
        
    # Append task
    with open(tasks_file, "a") as f:
        f.write(json.dumps(task) + "\n")
        
    print(f"✅ Injected goal: {goal}")
    return task

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject a new goal into the AGI system")
    parser.add_argument("goal", help="The goal to accomplish")
    parser.add_argument("--type", default="evolution", help="Goal type (default: evolution)")
    parser.add_argument("--priority", type=int, default=1, help="Priority 1-5 (default: 1)")
    
    args = parser.parse_args()
    inject_goal(args.goal, args.type, args.priority) 