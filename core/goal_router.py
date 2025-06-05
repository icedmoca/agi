from core.vector_memory import VectorMemory
from pathlib import Path
import json
import re
import logging
from typing import List, Dict, Set, Any
from core.utils import suggest_tags

logger = logging.getLogger(__name__)

class GoalRouter:
    # File-specific tag mappings
    FILE_TAG_MAPPINGS = {
        "core/memory.py": {"memory", "cache", "vector", "index", "storage"},
        "core/evolver.py": {"evolve", "refactor", "agent", "code", "generation"},
        "task_dashboard.py": {"ui", "dashboard", "stream", "visualization", "display"},
        "core/chat.py": {"chat", "llm", "model", "prompt", "message"},
        "core/agents/simulation.py": {"agent", "planner", "reviewer", "simulation"},
        "core/goal_gen.py": {"goal", "generation", "planning", "task"},
        "core/utils.py": {"utility", "helper", "tool", "common"},
        "core/self_heal.py": {"heal", "repair", "retry", "recovery", "resilience"}
    }
    
    def __init__(self):
        self.memory = VectorMemory()
        self.memory.load()
        
        # Load dependency map
        self.dep_map = {}
        map_path = Path("core/dependency_map.json")
        if map_path.exists():
            with open(map_path) as f:
                self.dep_map = json.load(f)
                
        self.file_patterns = self._load_file_patterns()
        
    def _load_file_patterns(self) -> Dict[str, Set[str]]:
        """Load and compile file-specific patterns"""
        patterns = {}
        for file_path in self.FILE_TAG_MAPPINGS.keys():
            if Path(file_path).exists():
                with open(file_path) as f:
                    content = f.read()
                    # Extract function and class names
                    identifiers = set(re.findall(r'(?:class|def)\s+(\w+)', content))
                    patterns[file_path] = identifiers
        return patterns
        
    def route_goal_to_files(self, goal: str, metadata: Dict[str, Any] = None) -> List[str]:
        """Route a goal to relevant files using tags and content analysis"""
        if metadata is None:
            metadata = {}
            
        # Direct routing for specific patterns
        if "retry" in goal.lower() and "self-heal" in goal.lower():
            return ["core/self_heal.py"]
            
        if "backoff" in goal.lower() and any(kw in goal.lower() for kw in ["retry", "heal", "repair"]):
            return ["core/self_heal.py"]
            
        # Get or generate tags
        tags = set(metadata.get("tags", suggest_tags(goal)))
        
        # Score each potential file
        file_scores = {}
        for file_path, relevant_tags in self.FILE_TAG_MAPPINGS.items():
            score = self._calculate_file_score(file_path, goal, tags, relevant_tags)
            if score > 0:
                file_scores[file_path] = score
                
        # Return top scoring files
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        return [file for file, score in sorted_files[:2]]  # Return top 2 most relevant files
        
    def _calculate_file_score(
        self, 
        file_path: str, 
        goal: str, 
        tags: Set[str], 
        relevant_tags: Set[str]
    ) -> float:
        """Calculate relevance score for a file"""
        score = 0.0
        
        # Tag overlap score
        tag_overlap = len(tags.intersection(relevant_tags))
        score += tag_overlap * 2.0
        
        # Pattern matching score
        if file_path in self.file_patterns:
            patterns = self.file_patterns[file_path]
            for pattern in patterns:
                if pattern.lower() in goal.lower():
                    score += 1.0
                    
        # File path relevance
        path_terms = set(file_path.lower().replace('/', ' ').replace('.', ' ').split())
        goal_terms = set(goal.lower().split())
        path_overlap = len(path_terms.intersection(goal_terms))
        score += path_overlap * 1.5
        
        return score
        
    def suggest_file_tags(self, file_path: str) -> Set[str]:
        """Suggest tags for a file based on its content and path"""
        return self.FILE_TAG_MAPPINGS.get(file_path, set())

    def find_related_files(self, goal: str) -> List[str]:
        """Find related files based on goal and memory"""
        # Get related memory entries
        related = self.memory.search(goal, top_k=3)
        
        # Extract file paths from memory entries
        matches = set()
        for entry in related:
            # Safely extract content from dict or fallback to string
            content = entry.get("content", "") if isinstance(entry, dict) else str(entry)
            file_matches = re.findall(r'core/[\w/]+\.py', content)
            matches.update(file_matches)
        
        # Add files based on keyword matching
        keywords = {
            r'evolve|change|modify|improve': ['core/evolver.py'],
            r'execute|run|action': ['core/executor.py'],
            r'watch|monitor|detect': ['core/watcher.py'],
            r'plan|decide|think': ['core/planner.py'],
            r'reflect|analyze': ['core/reflection.py']
        }
        
        return list(matches)