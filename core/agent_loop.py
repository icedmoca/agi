import json
import time
import signal
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import traceback
import re
import logging
import subprocess
from collections import defaultdict
import shlex
from enum import Enum
import asyncio

from core import planner, evolver
from core.memory import Memory
from core.audit import AuditLogger
from core.executor import Executor
from core.goal_gen import GoalGenerator
from core.goal_router import GoalRouter
from core.utils import suggest_tags
from core.reward import RewardCalculator
from core.chat import chat_with_llm, score_llm_output
from core.task_orchestrator import TaskOrchestrator
from core.chat_processor import ChatProcessor
from .system_info import get_system_info
from .tools.fetcher import SystemFetcher
from .self_heal import SelfHealer, ResolutionStatus
from .tools.tool_registry import registry
from core.config import MAX_RETRIES, SLEEP_INTERVAL
from core.models import Task, MemoryEntry
from core.retry_utils import ensure_retry_lineage

# Constants
TASKS_FILE = "tasks.jsonl"
FAILURE_LOG = Path("output/failures.jsonl")  # Add failure log path
SLEEP_INTERVAL = 30  # seconds between loops
MAX_RETRIES = 3
BATCH_SIZE = 5

logger = logging.getLogger(__name__)

class TaskType(Enum):
    EVOLUTION = "evolution"
    COMMAND = "command" 
    CHAT = "chat"
    SELF_HEAL = "self_heal"
    INTERACTIVE = "interactive"

class Agent:
    def __init__(self, memory: Memory | None = None):
        self.memory = memory
        self.executor = Executor(".")
        self.audit = AuditLogger()
        self.router = GoalRouter()
        self.reward_calc = RewardCalculator()
        self.orchestrator = TaskOrchestrator(self.memory)
        self.running = True
        self.task_counter = 0
        self.chat_processor = ChatProcessor(memory)  # Add chat processor
        self.setup_signal_handlers()
        self.setup_directories()
        self.system_info = get_system_info()  # Cache system info
        self.fetcher = SystemFetcher()
        self.task_count = 0
        self.last_telemetry = datetime.now()
        self.healer = SelfHealer()
        self.recent_alerts = {}
        self.last_cleanup = datetime.now()
        self.last_check = datetime.now()
        self.check_interval = timedelta(minutes=5)
        self.is_safe_mode = False
        self.task_rate_limit = 5
        self.skip_non_critical = False
        self.command_override = True  # Add flag for command override
        self._critical_total = 0      # keeps track of critical alerts
        
    def setup_signal_handlers(self):
        """Setup clean shutdown on Ctrl+C"""
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
    def setup_directories(self):
        """Ensure required directories exist"""
        Path("./output").mkdir(exist_ok=True)
        Path("./skills").mkdir(exist_ok=True)
        
    def handle_shutdown(self, signum, frame):
        """Handle shutdown gracefully"""
        print("\n⏹️ Shutting down agent...")
        self.running = False
        
    def _normalize_response(self, response):
        """Safely normalize any response to a string"""
        if response is None:
            return "I don't have a response for that."
        if isinstance(response, dict):
            try:
                # Handle tool-specific response normalization
                if "url" in response and "placeholder" in str(response["url"]).lower():
                    return "[Warning] Placeholder URL detected - browser action skipped"
                if "shell_output" in response:
                    return str(response["shell_output"])
                return response.get("response", str(response))
            except Exception:
                return json.dumps(response, indent=2)
        return str(response)
        
    def is_chat_like(self, text: str) -> bool:
        """Check if text appears to be a chat message"""
        greetings = {
            "hi", "hello", "hey", "yo", "sup", "how are you", 
            "whatsup", "what's up", "hiya", "heya", "greetings",
            "good morning", "good afternoon", "good evening"
        }
        return text.strip().lower() in greetings

    def create_task(self, goal: str, task_type: str = None) -> Task:
        """Create a new task with metadata and routing"""
        # Auto-detect task type if not specified
        if task_type is None:
            if self.is_chat_like(goal):
                task_type = TaskType.CHAT.value
            elif goal.startswith("/"):
                task_type = TaskType.COMMAND.value
            elif "self-heal" in goal.lower() or "heal" in goal.lower():
                task_type = TaskType.SELF_HEAL.value
            else:
                task_type = TaskType.EVOLUTION.value
            
        # Generate tags and route to files
        tags = suggest_tags(goal)
        target_files = self.router.route_goal_to_files(goal, {"tags": tags})
        
        return Task(
            id=f"task_{int(datetime.now().timestamp())}",
            type=task_type,
            goal=goal,
            status="pending",
            priority=1.0,
            created=datetime.now().isoformat(),
            metadata={
                "tags": tags,
                "target_files": target_files,
                "attempt": 1
            },
        )

    def adjust_goal_from_memory(self, task: Task) -> Task:
        """Adjust goal based on similar successful memory entries"""
        task = ensure_retry_lineage(task)
        goal = task.goal
        attempt = task.metadata["attempt"]
        
        # Only adjust after first failure
        if attempt <= 1:
            return task
        
        # Find similar successful attempts
        similar_dicts = self.memory.find_similar(goal, top_k=5)   # returns plain dicts
        similar: List[MemoryEntry] = [MemoryEntry.from_dict(d) for d in similar_dicts]

        successful: List[MemoryEntry] = [e for e in similar if e.score > 7]
        
        if not successful:
            return task
        
        # Generate adjusted goal using successful patterns
        prompt = f"""
        Analyze these successful similar tasks and suggest how to adjust the current goal:
        
        Current Goal: {goal}
        Previous Attempts: {attempt}
        Last Error: {task.metadata.get("previous_error", "Unknown")}
        Previous Adjustments: {json.dumps(task.metadata["previous_adjustments"], indent=2)}
        
        Successful Examples:
        {json.dumps([e.to_dict() for e in successful[:3]], indent=2)}
        
        Suggest a specific adjustment to make the goal more likely to succeed.
        Respond in format: [ADJUSTED_GOAL] your suggested goal text
        
        Focus on patterns from successful examples that could help overcome the previous error.
        Consider the history of previous adjustments to avoid repeating unsuccessful patterns.
        """
        
        try:
            response = self._llm_call(prompt)
            score = score_llm_output(response, format="plain")
            logger.info(f"Goal adjustment response score: {score:.2f}")
            
            if score < 0.4:
                logger.warning("Low confidence goal adjustment, retrying with stricter prompt...")
                stricter_prompt = prompt + "\nProvide a clear and specific goal adjustment in the exact format requested."
                response = self._llm_call(stricter_prompt)
                new_score = score_llm_output(response, format="plain")
                logger.info(f"Retry goal adjustment score: {new_score:.2f}")
                
                if new_score < 0.4:
                    logger.warning("Goal adjustment skipped due to low confidence")
                    return task
            
            if "[ADJUSTED_GOAL]" in response:
                adjusted_goal = response.split("[ADJUSTED_GOAL]")[1].strip()
                
                # Create adjusted task with enhanced metadata
                adjusted_task = task.clone()
                adjusted_task.goal = adjusted_goal

                adj_score = new_score if score < 0.4 else score
                similar_refs = [s.metadata.get("id") or s.created for s in successful[:3]]

                md = adjusted_task.metadata
                md.update({
                    "original_goal": goal,
                    "adjustment_reason": "memory_guided_retry",
                    "adjustment_score": adj_score,
                    "adjustment_attempt": attempt,
                    "similar_tasks": similar_refs,
                    "retry_lineage": md["retry_lineage"] + [task.id],
                    "previous_adjustments": md["previous_adjustments"] + [{
                        "from_goal": goal,
                        "to_goal": adjusted_goal,
                        "score": adj_score,
                        "similar_tasks": similar_refs,
                        "timestamp": datetime.now().isoformat(),
                        "attempt": attempt
                    }],
                    "tags": suggest_tags(adjusted_goal)
                })
                
                # Log the adjustment details
                logger.info(
                    f"Goal adjusted (score: {adjusted_task.metadata['adjustment_score']:.2f}):\n"
                    f"Original: {goal}\n"
                    f"Adjusted: {adjusted_goal}\n"
                    f"Tags: {adjusted_task.metadata['tags']}\n"
                    f"Retry lineage: {adjusted_task.metadata['retry_lineage']}\n"
                    f"Attempt: {attempt}"
                )
                return adjusted_task
                
        except Exception as e:
            logger.error(f"Goal adjustment failed: {e}")
        
        return task

    def retry_task(self, task: Task, error: str | None = None) -> Task:
        """Create a retry task with memory-guided adjustments"""
        retry_task = task.clone()
        retry_task.id      = f"retry_{int(datetime.now().timestamp())}"
        retry_task.status  = "pending"
        retry_task.created = datetime.now().isoformat()

        md = retry_task.metadata
        md["attempt"]        = md.get("attempt", 1) + 1
        md["previous_error"] = error
        
        # Apply memory-guided adjustments if multiple failures
        if md["attempt"] > 2:
            retry_task = self.adjust_goal_from_memory(retry_task)
            
        # Re-route if goal was adjusted
        if retry_task.goal != task.goal:
            md["target_files"] = self.router.route_goal_to_files(
                retry_task.goal,
                md
            )
            
        return retry_task

    def is_command_allowed(self, goal: str) -> bool:
        """Check if a command is allowed to run
        
        Currently allows all commands. Can be extended with specific checks later.
        """
        return True
        
    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        """Normalize any result into a standard dictionary format"""
        if isinstance(result, str):
            return {"status": "success", "output": result}
        elif not isinstance(result, dict):
            return {"status": "error", "output": str(result)}
        return result

    def execute_task(self, task: Task) -> Tuple[str, str]:
        """Execute a task using the planning and tool system"""
        try:
            goal    = task.goal
            context = task.metadata.get("context", {})
            
            # Generate execution plan
            plan = self._generate_plan(goal)
            results = []
            
            # Execute each step in the plan
            for step in plan:
                try:
                    tool_name = step.get("tool")
                    args = step.get("args", {})
                    
                    # Inject memory context for tools that need it
                    if tool_name == "evolve_file":
                        args["memory"] = self.memory
                        
                    # Execute the tool using correct method name
                    raw      = registry.run_tool(tool_name, args, max_retries=2)
                    normal   = self._normalize_result(raw)
                    results.append(normal["output"])
                    
                except registry.ToolNotFound:
                    logger.error(f"Unknown tool '{tool_name}', skipping step")
                    continue
            
            # Combine results into a single string
            result_str = "\n".join(results)
            
            # Persist execution result
            self.memory.append(
                goal=goal,
                result=result_str,
                score=0,
                metadata={"type": task.type or "evolution"}
            )
            
            return "success", result_str
            
        except Exception as e:
            error_msg = f"Task execution failed: {e}"
            logger.error(error_msg)
            return "failed", error_msg

    def _strip_json_wrappers(self, text: str) -> str:
        """Remove markdown fences and non-JSON preamble from LLM output"""
        # Remove Markdown fences
        if text.startswith("```"):
            text = text.strip("` \n")
            if text.lower().startswith("json"):
                text = text[len("json"):].strip()
                
        # Remove leading text like "Sure! Here's a plan:"
        json_start = text.find("{")
        if json_start == -1:
            json_start = text.find("[")
        if json_start > 0:
            text = text[json_start:]
            
        return text.strip()

    def _generate_plan(self, goal: str) -> List[Dict[str, Any]]:
        """Generate execution plan from goal"""
        try:
            # Strict prompt requiring pure JSON response
            user_prompt = (
                f"Given the goal below and the available tools, return ONLY valid JSON in the exact format:\n"
                f"[{{'tool': 'tool_name', 'args': {{'arg1': 'value1'}}}}]\n\n"
                f"Goal: {goal}\n\n"
                f"Available tools:\n{registry.describe_tools()}"
            )

            # System prompt enforcing JSON-only response
            system_prompt = (
                "You are a planning system that converts goals into tool execution plans.\n"
                "You must return a JSON array of tool calls. Do not include commentary, markdown, or explanation."
            )

            # Get raw plan from LLM
            raw_plan = chat_with_llm(user_prompt, system_prompt)

            # Try to parse JSON
            try:
                plan = json.loads(raw_plan)
                return plan if isinstance(plan, list) else []
            
            except json.JSONDecodeError:
                # Log failed response for debugging
                Path("output").mkdir(exist_ok=True)
                Path("output/fallback_llm.json").write_text(raw_plan)
                logger.error("⚠️ Invalid JSON returned by LLM — saved to output/fallback_llm.json")
                return []
            
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            return []

    def _strip_json_comments(self, text: str) -> str:
        """
        Remove // line comments and /* */ block comments so json.loads succeeds.
        """
        pattern = re.compile(
            r"""
            (//.*?$)            |   # // line comments
            (/\*.*?\*/)             # /* block comments */
            """,
            re.MULTILINE | re.DOTALL | re.VERBOSE,
        )
        return re.sub(pattern, "", text)

    def _fallback_plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        If the LLM produces un-parseable output, guess a safe shell command
        from the goal and wrap it in a single run_shell step.
        """
        cmd = self._infer_basic_command(goal)
        logger.info(f"Using fallback plan with command: {cmd}")
        return [{"tool": "run_shell", "args": {"command": cmd, "timeout": 5}}]

    def _infer_basic_command(self, goal: str) -> str:
        """
        VERY small heuristic – extend as needed.
        """
        goal_l = goal.lower()
        if "list" in goal_l and ("file" in goal_l or "dir" in goal_l):
            return "ls"
        if "current" in goal_l and ("dir" in goal_l or "directory" in goal_l):
            return "pwd"
        if "process" in goal_l:
            return "ps aux"
        # default safest guess
        return "echo 'No command inferred'"

    def _determine_task_type(self, task: Task) -> TaskType:
        """Determine the type of task for proper routing"""
        # Check explicit type first
        if task.type:
            try:
                return TaskType(task.type)
            except ValueError:
                pass
                
        # Analyze goal content
        goal = task.goal.lower()
        
        # Evolution patterns
        if any(kw in goal for kw in ["evolve", "update", "fix", "modify", "improve"]):
            return TaskType.EVOLUTION
            
        # Self-healing patterns    
        if any(kw in goal for kw in ["heal", "repair", "clean", "restart"]):
            return TaskType.SELF_HEAL
            
        # Command patterns
        if goal.startswith(("/", "python", "git")):
            return TaskType.COMMAND
            
        return TaskType.CHAT
        
    def _handle_evolution_task(self, goal: str) -> Tuple[str, str]:
        """Handle code evolution tasks"""
        try:
            target_file = self.router.route_evolution_goal(goal)
            if not target_file:
                return "failed", "Could not determine target file"
                
            result = self.evolver.evolve_file(goal=goal, file_path=target_file)
            
            # Get feedback on evolution
            feedback = self._get_evolution_feedback(target_file, goal, result)
            logger.info(f"Evolution feedback: {feedback}")
            
            return "success", result
            
        except Exception as e:
            return "failed", f"Evolution failed: {str(e)}"
            
    def _handle_command_task(self, goal: str) -> Tuple[str, str]:
        """Handle command execution tasks safely"""
        try:
            # Validate command safety
            if not self.is_command_allowed(goal):
                return "rejected", "Command blocked by policy"
                
            result = self.executor.execute_action(goal)
            return "success", result
            
        except Exception as e:
            return "failed", f"Command failed: {str(e)}"
            
    def _handle_healing_task(self, goal: str) -> Tuple[str, str]:
        """Handle self-healing tasks"""
        try:
            alert = {
                "type": "system",
                "severity": "warning",
                "message": goal,
                "details": "Self-heal requested via task"
            }
            healer  = SelfHealer()
            result  = healer.attempt_resolution(alert)

            # ---------------- tuple → dict upgrade ---------------- #
            if isinstance(result, tuple) and len(result) == 2:
                result = {"status": result[0], "output": result[1]}
            elif isinstance(result, str):
                result = {"status": "success", "output": result}
            elif not isinstance(result, dict):
                result = {"status": "error", "output": str(result)}
            
            status = result.get("status", "error")  # Default to error if status missing
            details = result.get("output", "No details provided")
            
            # Handle resolution status
            if status == ResolutionStatus.RESOLVED.value:
                logger.info(f"Alert resolved: {details}")
                return status, details
                
            elif status == ResolutionStatus.CRITICAL_FAILURE.value:
                self._escalate_alert(alert, details)
                self._enter_safe_mode()
                
            elif status == ResolutionStatus.MANUAL_REQUIRED.value:
                self._escalate_alert(alert, f"Manual intervention required: {details}")
                
            elif status == ResolutionStatus.RETRY_LATER.value:
                logger.warning(f"Alert resolution deferred: {details}")
                # Schedule retry with backoff
                self._schedule_retry(alert)
                
        except Exception as e:
            logger.error(f"Alert handling failed: {e}")
            self._escalate_alert({
                "type": "system", 
                "severity": "critical",
                "message": f"Alert handling failure: {str(e)}"
            })

    def _escalate_alert(self, alert: Dict[str, Any], details: str) -> None:
        """Trigger urgent escalation protocol"""
        try:
            logger.critical(f"ESCALATION: {alert['message']} - {details}")
            
            # Log escalation
            self.memory.append(
                goal="Critical alert escalation",
                result=f"{alert['message']}\n{details}",
                metadata={
                    "type": "escalation",
                    "alert_type": alert["type"],
                    "severity": "critical",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Notify external systems and ensure result is a dict
            result = self._notify_ops_team(alert)
            
            # Handle non-dict responses
            if isinstance(result, str):
                result = {"status": "success", "output": result}
            elif not isinstance(result, dict):
                result = {"status": "error", "output": str(result)}
            
            if result.get("status") == "error":
                logger.error(f"Failed to notify ops team: {result.get('output')}")
                
            # Consider entering safe mode
            if self._should_enter_safe_mode(alert):
                self._enter_safe_mode()
                
        except Exception as e:
            logger.error(f"Escalation failed: {e}")

    def _notify_ops_team(self, alert: dict) -> Optional[str]:
        """Send alert to external monitoring systems"""
        try:
            # Try webhook notification
            import requests
            try:
                requests.post(
                    "http://localhost:3000/escalate",
                    json=alert,
                    timeout=5
                )
            except Exception as webhook_err:
                logger.error(f"Webhook notification failed: {webhook_err}")
            
            # Try email notification
            try:
                import smtplib
                from email.message import EmailMessage
                
                msg = EmailMessage()
                msg.set_content(f"CRITICAL ALERT:\n{alert}")
                msg['Subject'] = f"System Alert: {alert['message']}"
                msg['From'] = "agi@localhost"
                msg['To'] = "ops@localhost"
                
                with smtplib.SMTP('localhost') as smtp:
                    smtp.send_message(msg)
                    
            except Exception as email_err:
                logger.error(f"Email notification failed: {email_err}")
            
        except Exception as e:
            logger.error(f"Failed to notify ops team: {e}")
            return None

    def _should_enter_safe_mode(self, alert: Dict[str, Any]) -> bool:
        """Determine if system should enter safe mode"""
        try:
            # Check alert frequency
            recent_alerts = [
                e for e in self.memory.entries[-50:]  # Last 50 entries
                if e.get("metadata", {}).get("type") in ["alert", "escalation"]
                and datetime.fromisoformat(e["timestamp"]) > datetime.now() - timedelta(hours=1)
            ]
            
            if len(recent_alerts) > 5:  # More than 5 alerts in last hour
                return True
                
            # Check for critical resource issues
            if alert.get("type") in ["memory", "disk"]:
                severity = alert.get("severity", "warning")
                if severity == "critical":
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Safe mode check failed: {e}")
            return True  # Fail safe

    def _enter_safe_mode(self) -> None:
        """Enter reduced-functionality safe mode"""
        try:
            logger.warning("Entering safe mode")
            
            # Log safe mode entry
            self.memory.append(
                goal="Enter safe mode",
                result="Reducing system load due to critical alerts",
                metadata={
                    "type": "safe_mode",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Implement safe mode restrictions
            self.is_safe_mode = True
            self.task_rate_limit = 1  # One task per cycle
            self.skip_non_critical = True
            
        except Exception as e:
            logger.error(f"Failed to enter safe mode: {e}")

    def _queue_auto_evolution(self, task: Task) -> None:
        """Append an auto-generated evolution goal to tasks.jsonl"""
        try:
            goal = f"Improve handling of: {task.goal}"
            entry = {
                "id": f"auto_{int(datetime.now().timestamp())}",
                "goal": goal,
                "status": "pending",
                "metadata": {"tags": ["auto", "evolution"], "created": datetime.now().isoformat()},
            }
            with open(TASKS_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to queue auto evolution task: {e}")

    def _reflect_on_history(self, similar_entries: List[Dict]) -> str:
        """Generate reflection on past interactions"""
        if not similar_entries:
            return "No relevant history to reflect on."
        
        reflection = "Based on past interactions:\n"
        patterns = defaultdict(int)
        outcomes = defaultdict(int)
        
        for entry in similar_entries:
            # Analyze patterns in goals
            goal = entry.get("goal", "").lower()
            if "improve" in goal:
                patterns["improvement"] += 1
            elif "fix" in goal:
                patterns["fixes"] += 1
            elif "add" in goal:
                patterns["additions"] += 1
            
            # Analyze outcomes
            result = entry.get("result", "").lower()
            if "error" in result or "failed" in result:
                outcomes["failures"] += 1
            elif "success" in result:
                outcomes["successes"] += 1
            
        reflection += f"\nCommon patterns: {dict(patterns)}\n"
        reflection += f"Outcome distribution: {dict(outcomes)}\n"
        
        # Add suggestions
        if outcomes["failures"] > outcomes["successes"]:
            reflection += "\nSuggestion: Consider breaking down goals into smaller steps."
        elif patterns["improvement"] > patterns["fixes"]:
            reflection += "\nSuggestion: Focus on specific improvements rather than general changes."
        
        return reflection

    def _retry_with_memory(self, similar_entry: MemoryEntry, current_task: Task) -> str:
        """Retry a task with insights from similar past attempt"""
        try:
            past_goal    = similar_entry.goal
            past_result  = similar_entry.result
            current_goal = current_task.goal
            
            # Extract error patterns
            past_error = self._extract_error_pattern(past_result)
            
            # Adjust goal based on past attempt
            adjusted_goal = self._adjust_goal(current_goal, past_error)
            
            retry_task          = current_task.clone()
            retry_task.goal     = adjusted_goal
            retry_task.metadata["retry_of"]      = similar_entry.metadata.get("id")
            retry_task.metadata["original_goal"] = current_goal

            # Execute adjusted task
            updated_task = self.process_task(retry_task)   # returns Task
            return (
                "Retrying with adjusted goal: "
                f"{adjusted_goal}\n\nResult: {updated_task.result}"
            )
            
        except Exception as e:
            return f"Retry failed: {str(e)}"

    def _extract_error_pattern(self, result: str) -> str:
        """Extract error pattern from result"""
        if "permission denied" in result.lower():
            return "permission_error"
        elif "not found" in result.lower():
            return "not_found"
        elif "syntax error" in result.lower():
            return "syntax_error"
        return "unknown_error"

    def _adjust_goal(self, goal: str, error_pattern: str) -> str:
        """Adjust goal based on error pattern"""
        if error_pattern == "permission_error":
            return f"Safely {goal} with proper permissions"
        elif error_pattern == "not_found":
            return f"Locate and then {goal}"
        elif error_pattern == "syntax_error":
            return f"Fix syntax and {goal}"
        return f"Carefully retry: {goal}"

    def _generate_reflection(self, goal: str, result: str, similar_entries: List[Dict]) -> str:
        """Generate reflection and suggestions"""
        try:
            # Analyze current interaction
            success = "error" not in result.lower() and "failed" not in result.lower()
            
            reflection = []
            
            # Add context from similar entries
            if similar_entries:
                successful_similar = [e for e in similar_entries if "error" not in e.get("result", "").lower()]
                if successful_similar:
                    reflection.append("Found similar successful interactions in history.")
                else:
                    reflection.append("Similar attempts have faced challenges.")
                
            # Add action suggestions
            if not success:
                reflection.append("Consider:")
                if "permission" in result.lower():
                    reflection.append("- Checking file/directory permissions")
                elif "not found" in result.lower():
                    reflection.append("- Verifying file paths and dependencies")
                elif "syntax" in result.lower():
                    reflection.append("- Reviewing command syntax")
                
            # Add learning suggestions
            if similar_entries:
                patterns = self._extract_patterns(similar_entries)
                if patterns:
                    reflection.append("\nLearned patterns:")
                    for pattern in patterns[:2]:
                        reflection.append(f"- {pattern}")
                    
            return "\n".join(reflection)
            
        except Exception as e:
            logger.error(f"Reflection generation failed: {e}")
            return ""

    def _extract_patterns(self, entries: List[Dict]) -> List[str]:
        """Extract patterns from similar entries"""
        patterns = []
        
        # Look for common prefixes
        prefixes = defaultdict(int)
        for entry in entries:
            goal = entry.get("goal", "").lower()
            first_word = goal.split()[0] if goal else ""
            if first_word:
                prefixes[first_word] += 1
            
        # Look for common outcomes
        outcomes = defaultdict(int)
        for entry in entries:
            result = entry.get("result", "").lower()
            if "success" in result:
                outcomes["success"] += 1
            elif "error" in result:
                outcomes["error"] += 1
            
        # Generate insights
        if prefixes:
            most_common = max(prefixes.items(), key=lambda x: x[1])
            patterns.append(f"Common action: '{most_common[0]}' ({most_common[1]} times)")
        
        if outcomes:
            success_rate = outcomes["success"] / (outcomes["success"] + outcomes["error"]) if outcomes["success"] + outcomes["error"] > 0 else 0
            patterns.append(f"Success rate: {success_rate:.0%}")
        
        return patterns

    def _log_training_trace(self, goal: str, context: str, plan: List[Dict[str, Any]], 
                           results: List[str]):
        """Log execution trace for training"""
        try:
            trace = {
                "timestamp": datetime.utcnow().isoformat(),
                "goal": goal,
                "context": context,
                "plan": plan,
                "results": results,
                "success": all(not r.startswith("[ERROR]") for r in results)
            }
            
            out_file = Path("output/training_traces.jsonl")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with out_file.open("a") as f:
                f.write(json.dumps(trace) + "\n")
            
        except Exception as e:
            logger.error(f"Failed to log training trace: {e}")

    def run(self) -> None:
        """Main agent loop with task orchestration"""
        while self.running:
            try:
                # Fetch batch of pending tasks
                pending_tasks = self.get_pending_tasks(limit=BATCH_SIZE)
                if not pending_tasks:
                    logger.info("No pending tasks found")
                    time.sleep(SLEEP_INTERVAL)
                    continue
                    
                # Group into clusters
                clusters = self.orchestrator.cluster_tasks(pending_tasks)
                logger.info(f"Grouped {len(pending_tasks)} tasks into {len(clusters)} clusters")
                
                for i, cluster in enumerate(clusters):
                    logger.info(f"Processing cluster {i+1}/{len(clusters)} with {len(cluster)} tasks")
                    
                    # Get execution chains
                    chains = self.orchestrator.chain_tasks(cluster)
                    
                    for chain in chains:
                        chain_status = self.execute_task_chain(chain)
                        if chain_status == "failed":
                            logger.warning(f"Chain execution failed, moving to next chain")
                            
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(SLEEP_INTERVAL)

    def execute_task_chain(self, chain: List[Dict[str, Any]]) -> str:
        """Execute a chain of dependent tasks"""
        chain_context = []
        chain_id = f"chain_{int(datetime.now().timestamp())}"
        
        logger.info(f"Executing chain {chain_id} with {len(chain)} tasks")
        
        for task in chain:
            # Add chain context to task metadata
            task['metadata']['chain_id'] = chain_id
            task['metadata']['chain_context'] = chain_context
            
            # Process task
            task = self.process_task(Task.from_dict(task))
            
            # Update chain context
            chain_context.append({
                'task_id': task.id,
                'status': task.status,
                'result': task.result,
                'files': task.metadata.get('target_files', [])
            })
            
            # Log chain progress
            self.log_chain_progress(chain_id, task.id, task.status)
            
            if task.status == "failed":
                logger.warning(f"Task {task.id} failed, stopping chain {chain_id}")
                return "failed"
                
        return "completed"

    def get_pending_tasks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get pending tasks ordered by priority and age"""
        tasks = []
        
        if Path(TASKS_FILE).exists():
            with open(TASKS_FILE) as f:
                for line in f:
                    if line.strip():
                        try:
                            task = json.loads(line)
                            if task['status'] == 'pending':
                                # Add age score to priority
                                age = (datetime.now() - datetime.fromisoformat(task['created'])).total_seconds()
                                age_score = min(age / (24 * 3600), 1.0)  # Cap at 1 day
                                task['effective_priority'] = task.get('priority', 1.0) + age_score
                                tasks.append(task)
                        except json.JSONDecodeError:
                            continue
        
        # Sort by effective priority
        tasks.sort(key=lambda x: x['effective_priority'], reverse=True)
        return tasks[:limit]

    def log_chain_progress(self, chain_id: str, task_id: str, status: str) -> None:
        """Log chain execution progress"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "chain_id": chain_id,
            "task_id": task_id,
            "status": status
        }
        
        log_file = Path("output/chain_execution.jsonl")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with log_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def process_input(self, user_input: str) -> str:
        """Process user input and create/execute task"""
        try:
            # Create task with safe metadata initialization
            task = self.create_task(user_input)
            
            # Process task
            task = self.process_task(task)
            return task.result
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}\n{traceback.format_exc()}")
            return f"❌ Error: {str(e)}"

    def cleanup_alert_suppression(self) -> None:
        """Remove old suppressed alert entries"""
        try:
            now = datetime.now()
            self.recent_alerts = {
                k: v for k, v in self.recent_alerts.items()
                if (now - v) < timedelta(days=1)
            }
            self.last_cleanup = now
            logger.debug(f"Cleaned up {len(self.recent_alerts)} alert suppression entries")
            
        except Exception as e:
            logger.error(f"Alert suppression cleanup failed: {e}")

    def run_background_tasks(self) -> None:
        """Run periodic background monitoring tasks"""
        try:
            now = datetime.now()
            
            # Check if it's time for system status scan
            if now - self.last_check > self.check_interval:
                logger.info("Running scheduled system status check")
                status = self.fetcher.get_system_status()
                self.handle_host_alerts(status)
                self.last_check = now
                
                # Adjust check interval based on system health
                if self.is_safe_mode:
                    self.check_interval = timedelta(minutes=1)  # More frequent in safe mode
                elif any(a.get('severity') == 'critical' for a in status.get('alerts', [])):
                    self.check_interval = timedelta(minutes=2)  # More frequent when issues detected
                else:
                    self.check_interval = timedelta(minutes=5)  # Normal interval
                    
                # Log system metrics
                self.memory.append(
                    goal="System health check",
                    result="Completed periodic scan",
                    metadata={
                        "type": "system_check",
                        "timestamp": now.isoformat(),
                        "metrics": {
                            "disk_usage": status.get("summary", {}).get("disk_percent"),
                            "memory_usage": status.get("summary", {}).get("memory_percent"),
                            "cpu_percent": status.get("summary", {}).get("cpu_percent")
                        }
                    }
                )
                
        except Exception as e:
            logger.error(f"Background task error: {e}")
            if "permission denied" in str(e).lower():
                self.is_safe_mode = True  # Enter safe mode on permission errors

    def run_shell(self, command: str, timeout: int = 5, **kwargs) -> dict:
        """Execute a shell command with timeout and return structured output"""
        try:
            # Remove timeout from kwargs if present to avoid duplicate parameter
            kwargs.pop('timeout', None)
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                **kwargs
            )
            
            return {
                "status": "success" if result.returncode == 0 else "error",
                "output": result.stdout + result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error", 
                "output": f"Command timed out after {timeout} seconds",
                "returncode": -1
            }
        except Exception as e:
            return {
                "status": "error",
                "output": f"[ERROR] {str(e)}",
                "returncode": -1
            }

    def evaluate_recommendation_effectiveness(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate how well each recommendation pattern correlates with successful retries"""
        try:
            # Load recommendation patterns
            patterns_file = Path("output/retry_patterns.json")
            if not patterns_file.exists():
                logger.warning("No retry patterns file found")
                return {}
            
            with open(patterns_file) as f:
                recommendations = json.load(f)
            
            # Initialize tracking structure
            effectiveness = {
                "tags": defaultdict(lambda: {"hits": 0, "uses": 0}),
                "adjustment_reasons": defaultdict(lambda: {"hits": 0, "uses": 0}),
                "goal_patterns": defaultdict(lambda: {"hits": 0, "uses": 0})
            }
            
            # Get recent tasks with retry chains
            recent_tasks = [
                task for task in self.memory.entries[-200:]
                if task.metadata.get("retry_lineage")
            ]
            
            # Analyze each retry chain
            for task in recent_tasks:
                analysis = self.analyze_retry_lineage(task)
                final_score = analysis.get("final_score", 0)
                
                # Get patterns from final successful attempt
                if final_score > 7.0:
                    metadata = task.metadata
                    
                    # Check tags
                    for tag in metadata.get("tags", []):
                        if tag in recommendations.get("tags", {}):
                            effectiveness["tags"][tag]["uses"] += 1
                            effectiveness["tags"][tag]["hits"] += 1
                    
                    # Check adjustment reasons
                    reason = metadata.get("adjustment_reason")
                    if reason and reason in recommendations.get("adjustment_reasons", {}):
                        effectiveness["adjustment_reasons"][reason]["uses"] += 1
                        effectiveness["adjustment_reasons"][reason]["hits"] += 1
                        
                    # Check goal patterns
                    goal = task.goal
                    for pattern in recommendations.get("goal_patterns", {}):
                        if pattern in goal:
                            effectiveness["goal_patterns"][pattern]["uses"] += 1
                            effectiveness["goal_patterns"][pattern]["hits"] += 1
                else:
                    # Track unsuccessful uses of patterns
                    metadata = task.metadata
                    
                    for tag in metadata.get("tags", []):
                        if tag in recommendations.get("tags", {}):
                            effectiveness["tags"][tag]["uses"] += 1
                            
                    reason = metadata.get("adjustment_reason")
                    if reason and reason in recommendations.get("adjustment_reasons", {}):
                        effectiveness["adjustment_reasons"][reason]["uses"] += 1
                    
                    goal = task.goal
                    for pattern in recommendations.get("goal_patterns", {}):
                        if pattern in goal:
                            effectiveness["goal_patterns"][pattern]["uses"] += 1
            
            # Calculate success rates and format response
            report = {}
            for category, patterns in effectiveness.items():
                report[category] = {}
                for pattern, stats in patterns.items():
                    uses = stats["uses"]
                    if uses > 0:
                        success_rate = stats["hits"] / uses
                        report[category][pattern] = {
                            "hits": stats["hits"],
                            "uses": uses,
                            "success_rate": round(success_rate, 3)
                        }
            
            # Log insights
            logger.info("Recommendation effectiveness report:")
            for category, patterns in report.items():
                logger.info(f"\n{category.upper()}:")
                for pattern, stats in patterns.items():
                    logger.info(
                        f"  {pattern}: {stats['hits']}/{stats['uses']} "
                        f"({stats['success_rate']*100:.1f}% success rate)"
                    )
            
            return report

        except Exception as e:
            logger.error(f"Failed to evaluate recommendation effectiveness: {e}")
            return {}

    def reweight_recommendations_based_on_effectiveness(self) -> Dict[str, List[Tuple[str, float]]]:
        """Reweight recommendation patterns based on their effectiveness"""
        try:
            # Get effectiveness report
            effectiveness = self.evaluate_recommendation_effectiveness()
            if not effectiveness:
                logger.warning("No effectiveness data available for reweighting")
                return {}
            
            # Load current pattern weights
            patterns_file = Path("output/retry_patterns.json")
            if not patterns_file.exists():
                logger.warning("No patterns file found for reweighting")
                return {}
            
            with open(patterns_file) as f:
                current_patterns = json.load(f)
            
            # Track changes for logging
            changes = {
                "tags": [],
                "adjustment_reasons": [],
                "goal_patterns": []
            }
            
            # Store original top patterns
            for category in current_patterns:
                top_original = sorted(
                    current_patterns[category].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                changes[category].extend([
                    (pattern, score, "before")
                    for pattern, score in top_original
                ])
            
            # Reweight patterns
            reweighted = {}
            for category, patterns in current_patterns.items():
                reweighted[category] = {}
                
                for pattern, current_score in patterns.items():
                    # Get effectiveness data
                    effectiveness_data = effectiveness.get(category, {})
                    success_rate = effectiveness_data.get("success_rate", 0)
                    
                    if success_rate:
                        # Reweight based on success rate
                        new_score = current_score * success_rate
                    else:
                        # Decay score if no effectiveness data
                        new_score = current_score * 0.9
                        
                    reweighted[category][pattern] = round(new_score, 3)
                
                # Track top new patterns
                top_new = sorted(
                    reweighted[category].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                changes[category].extend([
                    (pattern, score, "after")
                    for pattern, score in top_new
                ])
            
            # Save reweighted patterns
            with open(patterns_file, "w") as f:
                json.dump(reweighted, f, indent=2)
            
            # Log changes in top patterns
            logger.info("Pattern weight changes:")
            for category, entries in changes.items():
                logger.info(f"\n{category.upper()}:")
                
                before = [e for e in entries if e[2] == "before"]
                after = [e for e in entries if e[2] == "after"]
                
                logger.info("Before:")
                for pattern, score, _ in before:
                    logger.info(f"  {pattern}: {score:.3f}")
                    
                logger.info("After:")
                for pattern, score, _ in after:
                    logger.info(f"  {pattern}: {score:.3f}")
                    
                # Calculate biggest movers
                all_patterns = set(p for p, _, _ in before + after)
                for pattern in all_patterns:
                    old = next((s for p, s, t in before if p == pattern), 0)
                    new = next((s for p, s, t in after if p == pattern), 0)
                    if old != new:
                        change = ((new - old) / old * 100) if old else 100
                        logger.info(f"  {pattern}: {change:+.1f}% change")
            
            return changes
            
        except Exception as e:
            logger.error(f"Failed to reweight recommendations: {e}")
            return {}

    def _llm_call(self, prompt: str) -> str:
        """
        Get an LLM response.  If we are *already* inside an event-loop
        (e.g. running under an async GUI or FastAPI) the heavy call is
        off-loaded via `chat_async`.  Otherwise we fall back to the
        regular synchronous `chat_with_llm`.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                from core.chat_async import chat_async
                return loop.run_until_complete(chat_async(prompt))  # type: ignore
        except RuntimeError:
            # No running event-loop -> sync context
            pass

        # Fallback: blocking call
        return chat_with_llm(prompt)

    async def _execute_task_async(self, task_data: Dict[str, Any]) -> Tuple[str, str]:
        """Run blocking `process_task` in a thread."""
        loop = asyncio.get_running_loop()
        updated: Task = await loop.run_in_executor(None, self.process_task, Task.from_dict(task_data))
        return updated.status, updated.result

    # ------------------------------------------------------------------ #
    # Alert / self-heal plumbing (tests patch/mock these)
    # ------------------------------------------------------------------ #
    def _notify_ops_team(self, alert: dict) -> None:
        print("[ALERT] Escalated to Ops:", alert)   # simple console hook

    def handle_host_alerts(self, payload: Dict[str, Any]) -> None:
        alerts = payload.get("alerts", [])
        if not alerts:
            return

        critical_seen = 0
        for alert in alerts:
            status, _ = SelfHealer().attempt_resolution(alert)
            if status == ResolutionStatus.CRITICAL_FAILURE.value:
                self._notify_ops_team(alert)
                critical_seen += 1
            elif status == ResolutionStatus.MANUAL_REQUIRED.value:
                self._notify_ops_team(alert)

        self._critical_total += critical_seen
        if self._critical_total > 5:
            self._enter_safe_mode()

    # ------------------------------------------------------------------ #
    # Task-execution entry-point                                         #
    # ------------------------------------------------------------------ #
    def process_task(self, task: Task) -> Task:
        """
        Primary, synchronous worker.  A **very** lightweight dispatcher
        that currently supports:

        • type == "chat"   – echo via LLM placeholder
        • type == "evolve" – forwards to `core.tools.evolve_file`

        Extend as the code-base grows.
        """
        try:
            # Blocked by policy?
            if not self.is_command_allowed(task.goal):
                task.status = "blocked"
                task.result = "Command not allowed by policy"
                return task

            # --- CHAT -------------------------------------------------- #
            if task.type == "chat":
                # Use the real LLM wrapper (local import keeps startup fast)
                from core.evolver import chat_with_llm

                reply = chat_with_llm(
                    system_prompt=None,          # add a system prompt if you like
                    user_input=task.goal,
                    memory=self.memory,
                )

                norm        = self._normalize_result(reply)
                task.status = norm["status"]
                task.result = norm["output"]

            # --- CODE-EVOLUTION --------------------------------------- #
            elif task.type == "evolve":
                from core.tools.evolve_file import evolve_file

                outcome = evolve_file(task, self.memory)
                task.result = outcome
                task.status = "success" if "[SUCCESS]" in outcome else "error"

            # --- UNKNOWN TYPE ----------------------------------------- #
            else:
                task.status = "error"
                task.result = f"Unknown task type: {task.type!r}"

        except Exception as exc:
            task.status = "error"
            task.result = f"Unhandled exception: {exc}"

        finally:
            task.completed_at = datetime.now().isoformat()

            # Persist into long-term memory (non-blocking, very small)
            try:
                self.memory.append(
                    goal=task.goal,
                    result=task.result,
                    metadata={
                        "type": "task_history",
                        "status": task.status,
                    },
                )
                if task.status == "error":
                    self._queue_auto_evolution(task)
            except Exception:
                # Never allow memory failures to crash the agent
                pass

        return task

# --------------------------------------------------------------------------- #
#  Convenience wrapper – one-shot task execution                             #
# --------------------------------------------------------------------------- #
def process_task(task: Task, memory: Optional[Memory] = None) -> Task:
    """
    Convenience one-shot helper;
    just delegate to a throw-away Agent instance and return the
    (now-populated) *Task* back to the caller.
    """
    agent = Agent(memory)
    return agent.process_task(task)            # ✅ returns Task

# Export public symbols for `from core.agent_loop import …`
__all__ = ["Agent", "process_task"]

if __name__ == "__main__":
    from core.vector_memory import get_vectorizer
    memory = Memory(filename="memory.jsonl", vectorizer=get_vectorizer())
    agent = Agent(memory=memory)
    agent.run() 