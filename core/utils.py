import re
from typing import List, Set
from pathlib import Path
import json

# Common technical terms that make good tags
TECH_KEYWORDS = {
    'api', 'auth', 'cache', 'data', 'db', 'debug', 'doc', 'error', 'file', 
    'fix', 'log', 'memory', 'optimize', 'parser', 'refactor', 'test', 'ui',
    'validation', 'security', 'performance', 'agent', 'model', 'config',
    'stream', 'async', 'socket', 'route', 'middleware', 'schema', 'vector'
}

def suggest_tags(goal: str, max_tags: int = 3) -> List[str]:
    """Generate tags for a goal using keyword extraction"""
    # Convert to lowercase and extract words
    words = set(re.findall(r'\b\w+\b', goal.lower()))
    
    # Find technical keywords
    tech_tags = words.intersection(TECH_KEYWORDS)
    
    # Extract action-based tags
    action_words = {
        'fix': 'bugfix',
        'improve': 'enhancement',
        'optimize': 'optimization',
        'add': 'feature',
        'implement': 'feature',
        'refactor': 'refactor',
        'update': 'update',
        'test': 'testing'
    }
    
    action_tags = {
        action_words[word] 
        for word in words 
        if word in action_words
    }
    
    # Combine and limit tags
    all_tags = list(tech_tags.union(action_tags))
    return sorted(all_tags[:max_tags])

def update_task_tags(tasks_file: str = "tasks.jsonl") -> None:
    """Update existing tasks with auto-generated tags"""
    tasks_path = Path(tasks_file)
    if not tasks_path.exists():
        return
        
    tasks = []
    with tasks_path.open() as f:
        for line in f:
            if line.strip():
                try:
                    task = json.loads(line)
                    task["metadata"] = task.get("metadata", {})  # Safe initialization
                    if not task["metadata"].get("tags"):
                        task["metadata"]["tags"] = suggest_tags(task["goal"])
                    tasks.append(task)
                except json.JSONDecodeError:
                    continue
                    
    with tasks_path.open('w') as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n") 

def commit_changes(message: str) -> dict:
    """Commit changes to git repository"""
    try:
        import subprocess
        
        # Add all changes
        subprocess.run(["git", "add", "."], check=True)
        
        # Commit with message
        subprocess.run(["git", "commit", "-m", message], check=True)
        
        return {
            "status": "success",
            "output": f"Changes committed: {message}"
        }
    except Exception as e:
        return {
            "status": "error",
            "output": str(e)
        }

# ------------------------------------------------------------------ #
# Git Push helper – keeps repo in sync with remote
# ------------------------------------------------------------------ #

def push_changes(remote: str = "origin", branch: str = "HEAD") -> dict:
    """Push committed changes to the configured remote.

    Args:
        remote: Remote name (default "origin").
        branch: Branch or refspec to push (default current HEAD).

    Returns:
        Dict with *status* and *output* keys similar to other utils.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["git", "push", remote, branch],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            return {
                "status": "success",
                "output": result.stdout.strip() or "Pushed successfully",
            }
        return {
            "status": "error",
            "output": result.stderr.strip() or "Git push failed",
        }

    except Exception as e:
        return {
            "status": "error",
            "output": str(e),
        } 