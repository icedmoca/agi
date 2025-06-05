from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from core.memory import Memory
from core.evolver import Evolver
from core.goal_router import GoalRouter
from core.chat import chat_with_llm
from core.reward import score_evolution
from pathlib import Path
import json
from core.utils import suggest_tags
from core.models import Task

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.memory = Memory()
        
    def log_action(self, task_id: str, action: str, result: str) -> None:
        """Log agent action to agent_activity.jsonl"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "task_id": task_id,
            "action": action,
            "result": result
        }
        
        with open("output/agent_activity.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Planner", "Analyzes system state and proposes new goals")
        
    def run(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate new goals based on memory analysis"""
        memory_context = self.memory.get_recent_summary(limit=5)
        
        prompt = f"""
        As an AI system planner, analyze this context and propose 1-3 specific improvement goals:
        
        Recent Memory:
        {memory_context}
        
        Current Focus: {context.get('focus', 'general system improvement')}
        
        Propose goals in this format:
        1. <specific goal>
        2. <specific goal>
        3. <specific goal>
        """
        
        try:
            response = chat_with_llm(prompt)
            goals = self._parse_goals(response)
            
            for goal in goals:
                self.log_action(goal["id"], "proposed_goal", goal["goal"])
                
            return goals
        except Exception as e:
            logger.error(f"Planner error: {e}")
            return []
            
    def _parse_goals(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into goal dictionaries"""
        goals = []
        for line in response.split("\n"):
            if line.strip().startswith(("1.", "2.", "3.")):
                goal_text = line.split(".", 1)[1].strip()
                goals.append({
                    "id": f"goal_{int(datetime.now().timestamp())}",
                    "type": "evolution",
                    "goal": goal_text,
                    "priority": 1,
                    "status": "pending",
                    "created": datetime.now().isoformat(),
                    "metadata": {
                        "tags": suggest_tags(goal_text)
                    }
                })
        return goals
        
    def _generate_tags(self, goal: str) -> List[str]:
        """Generate tags for a goal using LLM"""
        prompt = f'Classify this goal with 1-3 lowercase tags (like bugfix, refactor, data, agent, io, experimental): "{goal}"'
        try:
            response = chat_with_llm(prompt)
            return [tag.strip() for tag in response.split(",")]
        except Exception:
            return ["untagged"]

class EvolverAgent(BaseAgent):
    def __init__(self):
        super().__init__("Evolver", "Implements code changes based on goals")
        self.router = GoalRouter()
        self.evolver = Evolver()
        
    def run(self, task: Dict[str, Any]) -> Tuple[str, str]:
        """Execute evolution for a task"""
        try:
            # Route goal to files
            files = self.router.route_goal_to_files(task["goal"])
            if not files:
                return "failed", "No relevant files found"
                
            # Evolve each file
            results = []
            for file in files:
                status, result = self.evolver.evolve_file(file, task["goal"])
                results.append(f"{file}: {result}")
                
            combined_result = "\n".join(results)
            self.log_action(task["id"], "evolution", combined_result)
            
            return "completed", combined_result
        except Exception as e:
            error_msg = f"Evolution error: {str(e)}"
            self.log_action(task["id"], "evolution_error", error_msg)
            return "failed", error_msg

class ReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Reviewer", "Evaluates evolution results")
        
    def run(self, task: Task, result: str) -> Dict[str, Any]:
        """Review evolution result and assign score"""
        try:
            # Calculate score
            score = score_evolution(result)
            
            # Generate review comment
            prompt = f"""
            Review this code evolution result and provide a brief assessment:
            
            Goal: {task.goal}
            Result: {result}
            Score: {score}
            
            Provide a 1-2 sentence review.
            """
            
            review = chat_with_llm(prompt)
            
            review_result = {
                "score": score,
                "review": review,
                "timestamp": datetime.now().isoformat()
            }
            
            self.log_action(task["id"], "review", json.dumps(review_result))
            return review_result
            
        except Exception as e:
            error_msg = f"Review error: {str(e)}"
            self.log_action(task["id"], "review_error", error_msg)
            return {"score": 0, "review": error_msg} 