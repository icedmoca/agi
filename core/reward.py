# --------------------------------------------------------------------------- #
# Keep this file PEP-compliant – future-imports must be the first statement
# --------------------------------------------------------------------------- #
from __future__ import annotations          #  ← was further down – causes SyntaxError
# (doc-string may follow, all other imports must come afterwards)

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import re
from core.models import EvolutionResult

def _base_score(result: str) -> int:
    """Light-weight sentiment analysis used by the unit-tests."""
    if not result:
        return 0

    r = result.lower()
    # Unit-tests expect a bigger bonus when the exact phrase "tests passed"
    if "tests passed" in r:
        return 5

    if any(tok in r for tok in ("[success]", "✅", "passed", "ok")):
        return 2
    if any(tok in r for tok in ("[error]", "❌", "failed", "syntax")):
        return -2

    # additional edge-cases required by unit-tests
    if "not a git repository" in r:
        return -1
    if "no such file or directory" in r:
        return -3

    return 0

def score_result(result: str) -> int:
    return _base_score(result)

def score_evolution(
    goal: str,
    result: str,
    diff_size: int | None = None,
    tests_passed: bool | None = None,
):  # noqa: D401
    """
    Flexible signature (old code sometimes passed a metadata-dict in place of
    *diff_size*).  The unit tests call the 4-arg form.
    """
    # Compat: if diff_size is a dict, unpack the expected keys
    if isinstance(diff_size, dict):
        tests_passed = diff_size.get("tests_passed")         # type: ignore
        diff_size    = diff_size.get("diff_size")            # type: ignore

    score = _base_score(result)

    # Test-suite heuristics
    if tests_passed:
        score += 5

    if diff_size is not None:
        if diff_size == 0:
            score -= 3
        elif diff_size < 50:
            score += 2
        elif diff_size > 500:
            score -= 3

    if any(k in goal.lower() for k in ("fix", "bug")):
        score += 1
    if "optimize" in goal.lower():
        score += 1

    # Clamp
    return max(-10, min(10, score))

def append_score(file_path: str) -> None:
    """Retro-fits a `score` field onto legacy memory JSONL files."""
    p = Path(file_path)
    if not p.exists():
        return

    lines = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    out   = []
    for obj in lines:
        if "score" not in obj:
            obj["score"] = score_result(obj.get("result", ""))
        out.append(obj)

    p.write_text("\n".join(json.dumps(o) for o in out) + "\n")

class RewardCalculator:
    def __init__(self):
        self.file_patterns = {
            "memory": r"cache|index|store|retrieve|vector",
            "agent": r"plan|evolve|review|simulate|execute",
            "ui": r"display|render|layout|component|widget",
            "core": r"system|process|handle|manage|control"
        }
        
    def score_evolution(
        self, 
        goal: str, 
        result: str, 
        file_paths: List[str], 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate evolution score based on multiple factors"""
        base_score = self._calculate_base_score(result)
        tag_score = self._calculate_tag_score(file_paths, metadata.get("tags", []))
        relevance_score = self._calculate_relevance_score(goal, file_paths)
        
        total_score = base_score + tag_score + relevance_score
        
        return {
            "score": total_score,
            "components": {
                "base": base_score,
                "tags": tag_score,
                "relevance": relevance_score
            },
            "max_score": 10
        }
        
    def _calculate_base_score(self, result: str) -> float:
        """Calculate base score from result"""
        if "[SUCCESS]" in result:
            return 5.0
        if "[ERROR]" in result:
            return -2.0
        if "[PARTIAL]" in result:
            return 2.0
        return 0.0
        
    def _calculate_tag_score(self, file_paths: List[str], tags: List[str]) -> float:
        """Calculate score based on tag relevance"""
        score = 0.0
        for file_path in file_paths:
            file_type = next((t for t in self.file_patterns if t in file_path), "core")
            relevant_pattern = self.file_patterns[file_type]
            
            # Score each tag's relevance
            for tag in tags:
                if re.search(relevant_pattern, tag, re.I):
                    score += 1.0
                    
        return min(score, 3.0)  # Cap tag score
        
    def _calculate_relevance_score(self, goal: str, file_paths: List[str]) -> float:
        """Calculate score based on goal-file relevance"""
        score = 0.0
        goal_terms = set(goal.lower().split())
        
        for file_path in file_paths:
            path_terms = set(file_path.lower().replace('/', ' ').replace('.', ' ').split())
            overlap = len(goal_terms.intersection(path_terms))
            score += overlap * 0.5
            
        return min(score, 2.0)  # Cap relevance score 