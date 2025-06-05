import os
import json
import time
import sys
import subprocess
import traceback
from datetime import datetime
import ollama
import pathlib
import re
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from logging import getLogger, Formatter, ERROR, INFO, StreamHandler
from faiss import IndexFlatL2
from hashlib import sha256
import logging
from pathlib import Path
import shlex

# --- Shell command validation helpers ------------------------------------

VALID_PREFIXES = {
    "cd", "ls", "echo", "cat", "touch", "mkdir", "rm", "python", "pip",
    "grep", "find", "pwd", "cp", "mv", "pytest", "git", "sed", "awk",
    "head", "tail", "du", "df", "chmod", "chown", "for", "while", "curl",
    "wget",
}


def is_valid_shell_command(cmd: str) -> bool:
    """Return True if the string looks like a real shell command."""
    cmd = cmd.strip()
    if not cmd:
        return False

    # Obvious natural language lines like "And then run" or "Make sure".
    if re.match(r"^[A-Za-z][A-Za-z\s]{3,}$", cmd) and not re.search(r"[;&|$]", cmd):
        return False

    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return False

    if not tokens:
        return False

    first = tokens[0]
    if first.startswith(('./', '/')) or first.endswith('.sh'):
        return True

    first_l = first.lower()
    if first_l in VALID_PREFIXES:
        return True

    # Allow simple control structures
    if first_l in {"if", "for", "while"}:
        return True

    return False


def is_likely_shell_command(cmd: str) -> bool:
    """Backward compatible alias for is_valid_shell_command."""
    return is_valid_shell_command(cmd)

# Configure module logger
logger = logging.getLogger(__name__)

LOGGER = getLogger(__name__)
LOG_FORMATTER = Formatter(f"[%(asctime)s] %(levelname)s: %(message)s")
LOG_STREAM_HANDLER = StreamHandler()
LOG_STREAM_HANDLER.setFormatter(LOG_FORMATTER)
LOGGER.addHandler(LOG_STREAM_HANDLER)
LOGGER.setLevel(ERROR)

class Executor:
    def __init__(self, working_dir: str = "."):
        self.working_dir = Path(working_dir)
        self.timeout = 60  # Default timeout in seconds
        self.max_output = 1024 * 1024  # 1MB output limit
        self.valid_prefixes = {
            "cd", "ls", "echo", "cat", "grep", "find", "pwd", "touch",
            "mkdir", "rm", "cp", "mv", "python", "pytest", "pip", "git",
            "sed", "awk", "head", "tail", "du", "df", "chmod", "chown",
            "for", "while", "if", "bash", "sh", "curl", "wget",
        }

    def _is_shell_command(self, command: str) -> bool:
        """Simple heuristic to determine if text looks like a shell command"""
        return is_valid_shell_command(command)
        
    def run_command(self, command: str) -> Tuple[int, str, str]:
        """Execute a shell command and return (returncode, stdout, stderr)"""
        try:
            # Sanitize and parse command
            command = command.strip()
            if command.startswith('$'):
                command = command[1:].strip()

            if not is_valid_shell_command(command):
                logger.warning("Skipped invalid shell command: %s", command)
                return 1, "", "Rejected non-shell command"

            args = shlex.split(command)
            
            # Execute command with proper error handling
            process = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir
            )
            
            return (
                process.returncode,
                process.stdout[:self.max_output],
                process.stderr[:self.max_output]
            )
            
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {self.timeout} seconds"
        except subprocess.SubprocessError as e:
            return 1, "", f"Command failed: {str(e)}"
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return 1, "", f"Error: {str(e)}"
            
    def execute_task(self, task: Dict[str, Any]) -> Tuple[str, str]:
        """Execute a task and return (status, result)"""
        try:
            goal = task.get("goal", "")
            task_type = task.get("type", "command")
            
            if task_type == "command":
                # Execute shell command
                returncode, stdout, stderr = self.run_command(goal)
                
                if returncode == 0:
                    result = stdout if stdout else "Command completed successfully"
                    status = "completed"
                else:
                    result = f"Command failed:\n{stderr if stderr else stdout}"
                    status = "failed"
                    
            else:
                # Handle other task types
                result = f"Unsupported task type: {task_type}"
                status = "failed"
                
            return status, result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return "failed", str(e)

    def _cache_key(self, goal: str, result: str) -> str:
        """Generate cache key from goal and result"""
        return sha256(f"{goal}::{result}".encode()).hexdigest()
        
    def _cache_update(self, entry: Dict[str, Any]) -> None:
        """Update cache with new entry"""
        key = self._cache_key(entry["goal"], entry["result"])
        self.cache[key] = entry
        
        # Limit cache size using FIFO
        if len(self.cache) > self.cache_size:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            
    def _build_index(self) -> None:
        """Build FAISS index from vectors"""
        if not self.vectors or not self.vector_dim:
            return
            
        self.index = IndexFlatL2(self.vector_dim)
        vectors_array = np.array(self.vectors).astype('float32')
        self.index.add(vectors_array)

    def run_and_capture(self, cmd: Union[List[str], str], **kwargs) -> subprocess.CompletedProcess:
        """Run a command and capture output safely"""
        try:
            shell_cmd = cmd if isinstance(cmd, str) else ' '.join(cmd)
            if not shell_cmd or not is_valid_shell_command(shell_cmd):
                return subprocess.CompletedProcess(cmd, -1, stdout="", stderr="[ERROR] Rejected non-shell command")

            result = subprocess.run(
                cmd,
                shell=isinstance(cmd, str),
                capture_output=True,
                text=True,
                timeout=30,
                **kwargs
            )
            if result.returncode != 0 or result.stderr:
                self._reflect_failure(shell_cmd, result.stderr or result.stdout)
            return result
        except subprocess.TimeoutExpired:
            self._reflect_failure(shell_cmd if isinstance(cmd, str) else ' '.join(cmd), "Command timed out")
            return subprocess.CompletedProcess(
                cmd, -1, stdout="", stderr="[ERROR] Command timed out"
            )
        except Exception as e:
            self._reflect_failure(shell_cmd if isinstance(cmd, str) else ' '.join(cmd), str(e))
            return subprocess.CompletedProcess(
                cmd, -1, stdout="", stderr=f"[ERROR] {str(e)}"
            )

    def _reflect_failure(self, cmd: str, error: str) -> None:
        """Use the LLM to reflect on failed shell commands and log to memory"""
        try:
            from core.chat import chat_with_llm
            from core.memory import Memory
            from core.reward import score_result

            prompt = (
                "This shell command failed:\n\n"
                f"{cmd}\n\n"
                f"Error:\n{error}\n\n"
                "Why did it fail? Suggest a safer alternative."
            )
            reflection = chat_with_llm(prompt)
            Memory().append(
                goal=cmd,
                result=reflection,
                score=score_result(reflection),
                metadata={"error": error, "timestamp": datetime.now().isoformat()},
            )
            logger.info("Reflection saved for failed command: %s", cmd)
        except Exception as exc:
            logger.error("Failed to record reflection: %s", exc)

    def execute_action(self, goal_description: str) -> str:
        """Execute an action safely with memory and fallbacks"""
        try:
            # Sanitize command
            command = self.sanitize_command(goal_description)

            from core.intent_classifier import classify_intent

            outputs = []
            executed = 0
            skipped = 0

            for line in command.splitlines():
                if not line.strip():
                    continue
                cls = classify_intent(line)
                ctype = cls.get("type", "other")
                cleaned = cls.get("value", line).strip()
                LOGGER.info("Command classification: %s - %s", ctype, cleaned)
                if ctype == "interactive":
                    reason = cls.get("reason", "Blocks on user input")
                    LOGGER.warning(f"Skipped interactive command: {cleaned} — {reason}")
                    skipped += 1
                    continue
                if ctype != "command":
                    LOGGER.warning("Skipped non-command: %s - %s", ctype, cleaned)
                    skipped += 1
                    continue
                if not is_valid_shell_command(cleaned):
                    LOGGER.warning("Rejected non-shell command: %s", cleaned)
                    skipped += 1
                    continue

                if cleaned.lower().startswith("evolve ") and " to " in cleaned:
                    return self._handle_evolution(cleaned)

                result = self.run_and_capture(
                    cleaned,
                    cwd=self.working_dir
                )
                executed += 1
                if result.returncode != 0:
                    return f"[ERROR] Return code {result.returncode}\n{result.stderr}"
                outputs.append(result.stdout)

            logger.debug("Executed %d commands, skipped %d", executed, skipped)
            if executed == 0:
                return "[SKIPPED]"
            return "\n".join(filter(None, outputs)) or "[SUCCESS] No output"
            
        except Exception as e:
            error_msg = f"[ERROR] {str(e)}\n{traceback.format_exc()}"
            LOGGER.error(error_msg)
            return error_msg
            
    def _handle_evolution(self, command: str) -> str:
        """Handle evolution request safely"""
        try:
            parts = command.split(" to ", 1)
            filename = parts[0].replace("evolve ", "").strip()
            improvement = parts[1].strip()
            
            if not filename.endswith('.py'):
                return "[ERROR] Only Python files can be evolved"
                
            file_path = f"{self.working_dir}/{filename}"
            if not os.path.isfile(file_path):
                return "[ERROR] File not found or unsafe request"
                
            from core.evolver import evolve_file
            result = evolve_file(goal=improvement, file_path=file_path)
            
            # Clean generated code before returning
            result = self.clean_generated_code(result)
            
            # Update memory
            from core.memory import Memory
            mem = Memory()
            mem.append(f"evolve {filename}", result)
            
            return result
            
        except Exception as e:
            return f"[ERROR] Evolution failed: {str(e)}"

    def sanitize_command(self, cmd: str) -> str:
        """Sanitize commands by removing unsafe patterns"""
        cmd = re.sub(r'[`;&|]', '', cmd)
        return cmd.strip()

    def main(self) -> None:
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            self.run_tests()

    def run_tests(self):
        execute_action("run tests")

    @staticmethod
    def get_valid_commands() -> Dict[str, Tuple[str, Dict]]:
        """Return a dictionary of valid commands and their metadata"""
        return {k: v for k, v in Executor().VALID_COMMANDS.items()}

    def execute_evolution(self, goal: str, file_path: str) -> str:
        """Execute an evolution task"""
        try:
            # Validate inputs
            if not goal or not file_path:
                return "[ERROR] Missing goal or file path"
            
            logger.info(f"🎯 Processing evolution request: {goal}")
            
            # Check if auto-approve
            auto_approve = any(goal.lower().startswith(prefix) for prefix in 
                             ("fix", "update", "add logging"))
            
            if not auto_approve and not self.confirm_evolution(goal, file_path):
                return "[CANCELLED] Evolution cancelled by user"
            
            # Execute evolution
            result = evolver.evolve_file(goal, file_path)
            
            # Score and log
            score = score_evolution(goal=goal, result=result)
            logger.info(f"📊 Evolution score: {score}")
            
            if score < 0:
                logger.warning("⚠️ Low evolution score, might need manual review")
            
            return result
            
        except Exception as e:
            error_msg = f"[ERROR] Evolution failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return error_msg

    def clean_generated_code(self, text: str) -> str:
        """Clean and sanitize generated code output"""
        # Strip markdown code fences and language tags
        text = re.sub(r"^```(?:python)?", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"```$", "", text, flags=re.MULTILINE)

        # Strip invisible characters and weird symbols 
        text = ''.join(c for c in text if c.isprintable())

        # Remove repeated shebangs or encoding lines
        lines = text.splitlines()
        clean_lines = []
        for line in lines:
            if re.match(r'^#!', line) or 'coding' in line:
                continue
            clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()

# Global function for backwards compatibility
def execute_action(command: str) -> str:
    """Run a shell command and return its output"""
    try:
        command = re.sub(r'[`;&|]', '', command).strip()

        from core.intent_classifier import classify_intent
        exec_ = Executor()

        outputs = []
        executed = 0
        skipped = 0

        for line in command.splitlines():
            if not line.strip():
                continue
            cls = classify_intent(line)
            ctype = cls.get("type", "other")
            cleaned = cls.get("value", line).strip()
            LOGGER.info("Command classification: %s - %s", ctype, cleaned)
            if ctype == "interactive":
                reason = cls.get("reason", "Blocks on user input")
                LOGGER.warning(f"Skipped interactive command: {cleaned} — {reason}")
                skipped += 1
                continue
            if ctype != "command":
                LOGGER.warning("Skipped non-command: %s - %s", ctype, cleaned)
                skipped += 1
                continue
            if not is_valid_shell_command(cleaned):
                LOGGER.warning("Skipped invalid shell command: %s", cleaned)
                skipped += 1
                continue

            if cleaned.lower().startswith("evolve ") and " to " in cleaned:
                try:
                    parts = cleaned.split(" to ", 1)
                    filename = parts[0].replace("evolve ", "").strip()
                    improvement = parts[1].strip()
                    if not filename.endswith('.py'):
                        return "[ERROR] Only Python files can be evolved"
                    file_path = f"core/{filename}"
                    if not os.path.isfile(file_path):
                        return "[ERROR] File not found or unsafe request"
                    from core.evolver import evolve_file
                    result = evolve_file(goal=improvement, file_path=file_path)
                    result = clean_generated_code(result)
                    from core.memory import Memory
                    mem = Memory()
                    mem.append(f"evolve {filename}", result)
                    recent = mem.get_recent(5)
                    if recent:
                        print("\n📚 Recent Evolution Context:")
                        for entry in recent:
                            score = entry.get("score", "?")
                            print(f"[MEMORY] {entry['goal']} ➜ {entry['result']} (score: {score})")
                        print()
                    return result
                except Exception as e:
                    error_msg = f"[ERROR] Evolution failed: {str(e)}"
                    from core.memory import Memory
                    Memory().append(cleaned, error_msg)
                    return error_msg
                continue

            proc = exec_.run_and_capture(
                cleaned,
                cwd=exec_.working_dir
            )
            executed += 1
            if proc.returncode != 0:
                return f"[ERROR] Return code: {proc.returncode}\n{proc.stderr}"
            outputs.append(proc.stdout.strip())

        logger.debug("Executed %d commands, skipped %d", executed, skipped)
        final_result = "\n".join(filter(None, outputs)) or "[SUCCESS] No output"

        from core.memory import Memory
        mem = Memory()
        mem.append(command, final_result)

        recent = mem.get_recent(5)
        if recent:
            print("\n📚 Recent Memory Context:")
            for entry in recent:
                score = entry.get("score", "?")
                print(f"[MEMORY] {entry['goal']} ➜ {entry['result']} (score: {score})")
            print()

        return final_result
        
    except Exception as e:
        error_msg = f"[ERROR] Exception: {str(e)}"
        # Log error to memory
        from core.memory import Memory
        Memory().append(command, error_msg)
        return error_msg

def clean_generated_code(text: str) -> str:
    """Clean and sanitize generated code output"""
    import re
    
    # Strip markdown code fences and language tags
    text = re.sub(r"^```(?:python)?", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)

    # Strip invisible characters and weird symbols 
    text = ''.join(c for c in text if c.isprintable())

    # Remove repeated shebangs or encoding lines
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        if re.match(r'^#!', line) or 'coding' in line:
            continue
        clean_lines.append(line)
        
    return '\n'.join(clean_lines).strip()

if __name__ == "__main__":
    executor = Executor(os.getenv("PROJECT_ROOT", "/home/astro/agi"))
    executor.main()