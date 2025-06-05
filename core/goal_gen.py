import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import re
import random
import time

from core.memory import Memory
from core.vector_memory import VectorMemory
from core.models import MemoryEntry

logger = logging.getLogger(__name__)

class Goal:
    def __init__(self, goal_type: str, description: str, parent_id: Optional[str] = None, source: Optional[str] = None):
        self.id = f"goal_{int(time.time())}_{random.randint(1000,9999)}"
        self.type = goal_type
        self.description = description
        self.parent_id = parent_id
        self.source = source
        self.created_at = datetime.now().isoformat()
        self.status = "pending"
        self.result = None
        self.score = 0
        self.metadata = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary"""
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "parent_id": self.parent_id,
            "source": self.source,
            "created_at": self.created_at,
            "status": self.status,
            "result": self.result,
            "score": self.score,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        """Create goal from dictionary"""
        goal = cls(
            goal_type=data["type"],
            description=data["description"],
            parent_id=data.get("parent_id"),
            source=data.get("source")
        )
        goal.id = data["id"]
        goal.created_at = data["created_at"]
        goal.status = data["status"]
        goal.result = data.get("result")
        goal.score = data.get("score", 0)
        goal.metadata = data.get("metadata", {})
        return goal

class GoalGenerator:
    def __init__(self, memory_file: str = "memory.jsonl", tasks_file: str = "tasks.jsonl"):
        self.memory = Memory(memory_file)
        self.vector_memory = VectorMemory()
        self.tasks_file = Path(tasks_file)
        self.improvement_types = [
            "improve error handling",
            "optimize performance",
            "enhance input validation",
            "reduce memory usage",
            "increase code clarity",
            "strengthen safety checks",
            "add better logging",
            "improve documentation"
        ]
        
    def generate_goals(self, lookback_hours: int = 24) -> List[Dict[str, Any]]:
        """Generate new goals based on memory analysis"""
        logger.info("🧠 Analyzing memory for goal generation...")
        
        # Analyze memory patterns
        failures = self._find_failure_patterns()
        low_scores = self._find_low_scoring_goals()
        retry_candidates = self._find_retry_candidates()
        
        # Generate goals for each pattern
        goals = []
        goals.extend(self._generate_failure_fixes(failures))
        goals.extend(self._generate_improvement_goals(low_scores))
        goals.extend(self._generate_retry_goals(retry_candidates))
        
        # Write goals to tasks file
        self._write_goals(goals)
        
        return goals
        
    def _find_failure_patterns(self) -> Dict[str, List[MemoryEntry]]:
        """Find patterns in failed evolutions"""
        patterns = defaultdict(list)
        
        for entry in self.memory.entries:
            if "[ERROR]" in entry.result:
                error_type = self._categorize_error(entry.result)
                patterns[error_type].append(entry)
                
        return patterns
        
    def _find_low_scoring_goals(self) -> List[MemoryEntry]:
        """Find goals with consistently low scores"""
        return [
            entry for entry in self.memory.entries
            if entry.score < 5
        ]
        
    def _find_retry_candidates(self) -> List[MemoryEntry]:
        """Find goals that failed multiple times"""
        return [
            e for e in self.memory.entries
            if not e.metadata.get("retry_of")
        ]
        
    def _generate_failure_fixes(self, failures: Dict[str, List[MemoryEntry]]) -> List[Dict]:
        """Create goals that directly address recurring failures"""
        goals: List[Dict] = []
        for err_type, entries in failures.items():
            for entry in entries:
                goal_txt = f"Fix {err_type}: {entry.goal}"
                
                goals.append(
                    {
                        "id": f"fix_{int(time.time()*1000)}",
                        "type": "evolution",
                        "goal": goal_txt,
                        "priority": 2,
                        "metadata": {
                            "original_goal": entry.goal,
                            "last_error": self._extract_error(entry.result),
                            "error_type": err_type,
                            "original_score": entry.score,
                        },
                    }
                )
        return goals
        
    def _generate_improvement_goals(self, low_scores: List[MemoryEntry]) -> List[Dict]:
        """Suggest goals that improve poorly-scoring attempts"""
        goals: List[Dict] = []
        for entry in low_scores:
            similar = self.vector_memory.search(entry.goal, top_k=3)
            target_files = self._extract_target_files(similar)
            
            goals.append(
                {
                    "id": f"improve_{int(time.time()*1000)}",
                    "type": "evolution",
                    "goal": f"Improve {self._extract_target(entry.goal)}",
                    "priority": 2,
                    "metadata": {
                        "original_goal": entry.goal,
                        "original_score": entry.score,
                        "target_files": target_files,
                        "reason": "low_score",
                    },
                }
            )
        return goals
        
    def _generate_retry_goals(self, retry_candidates: List[MemoryEntry]) -> List[Dict]:
        """Propose retries for tasks that failed repeatedly but were never retried"""
        goals: List[Dict] = []
        for entry in retry_candidates:
            goals.append(
                {
                    "id": f"retry_{int(time.time()*1000)}",
                    "type": "evolution",
                    "goal": f"Retry with alternative approach: {entry.goal}",
                    "priority": 3,
                    "metadata": {
                        "reason": "multiple_failures",
                        "original_goal": entry.goal,
                        "attempt_count": self._count_attempts(entry.goal),
                        "last_error": self._extract_error(entry.result),
                    },
                }
            )
        return goals
        
    def _write_goals(self, goals: List[Dict]) -> None:
        """Write generated goals to tasks file"""
        for goal in goals:
            goal["created"] = datetime.now().isoformat()
            goal["status"] = "pending"
            
            with self.tasks_file.open("a") as f:
                f.write(json.dumps(goal) + "\n")
                
    def _categorize_error(self, error: str) -> str:
        """Categorize error message into type"""
        patterns = {
            "syntax": r"SyntaxError|IndentationError",
            "import": r"ImportError|ModuleNotFoundError",
            "attribute": r"AttributeError",
            "type": r"TypeError",
            "validation": r"ValidationError|AssertionError",
            "io": r"FileNotFoundError|PermissionError"
        }
        
        for category, pattern in patterns.items():
            if re.search(pattern, str(error)):
                return category
        return "unknown"
        
    def _extract_affected_files(self, entries: List[MemoryEntry]) -> List[str]:
        """Extract unique affected files from entries"""
        files = set()
        for entry in entries:
            matches = re.findall(r'core/[\w/]+\.py', str(entry))
            files.update(matches)
        return list(files)
        
    def _extract_target(self, goal: str) -> str:
        """Extract the target of improvement from goal"""
        # Remove common prefixes
        cleaned = re.sub(r'^(fix|update|improve|modify)\s+', '', goal.lower())
        return cleaned
        
    def _count_attempts(self, goal: str) -> int:
        """Count number of attempts for a goal"""
        return sum(1 for e in self.memory.entries if e.goal == goal)
        
    def _extract_error(self, result: str) -> str:
        """Extract error message from result"""
        if isinstance(result, str):
            error_match = re.search(r'\[ERROR\]\s*(.+)', result)
            if error_match:
                return error_match.group(1)
        return str(result)

    def generate_goal(self, context: str) -> Goal:
        """Generate a new goal based on context"""
        goal_type = random.choice(self.improvement_types)
        return Goal(
            goal_type=goal_type,
            description=f"{goal_type}: {context}"
        ) 