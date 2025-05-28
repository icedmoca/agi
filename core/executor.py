python
import os
import json
import time
import sys
import subprocess
from datetime import datetime
import ollama
import pathlib
from typing import Tuple, Optional, List, Dict
from logging import getLogger, Formatter, ERROR, INFO
EXECUTOR_MODULES = __import__("core.executor_modules").ExecutorsModule  # Importing the module with necessary functions
LOGGER = getLogger(__name__)
LOG_FORMATTER = Formatter(f"[%(asctime)s] %(levelname)s: %(message)s")
LOG_STREAM_HANDLER = StreamHandler()
LOG_STREAM_HANDLER.setFormatter(LOG_FORMATTER)
LOGGER.addHandler(LOG_STREAM_HANDLER)
LOGGER.setLevel(ERROR)
class Executor:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.bad_patterns_file = f"{project_root}/output/bad_patterns.jsonl"
        self.fallback_executor = f"{project_root}/core/executor.py"
    def sanitize_command(self, cmd: str) -> str:
        """Sanitize commands by removing unsafe patterns"""
        cmd = re.sub(r'[`;&|]', '', cmd)
        return cmd.strip()
    def main(self) -> None:
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            self.run_tests()
    def run_tests(self):
        execute_action("run tests")
    def execute_action(self, goal_description: str) -> str:
        """Top-level action executor for AGI goals"""
        executor = EXECUTOR_MODULES()
        if goal_description.startswith("evolve "):
            try:
                # Extract file and evolution goal
                parts = goal_description.split(" to ")
                filename = parts[0].replace("evolve ", "").strip()
                improvement = parts[1].strip()
                # Validate file is in core directory
                if not filename.endswith('.py'):
                    return "[ERROR] Only Python files can be evolved"
                file_path = f"{self.project_root}/core/{filename}"
                if not os.path.isfile(file_path):
                    return "[ERROR] File not found or unsafe request"
                # Call evolver with extracted parameters
                print(f" Evolution Request:\nFile: {filename}\nGoal: {improvement}")
                return executor.evolve_file(goal=improvement, file_path=file_path)
            except Exception as e:
                return f"[ERROR] Evolution failed: {str(e)}"
        elif goal_description in executor.VALID_COMMANDS:
            command = executor.VALID_COMMANDS[goal_description][0]
            return executor.execute_command(command, executor.VALID_COMMANDS[goal_description][1])
        else:
            LOGGER.error("Unsupported action type")
            return "[ERROR] Unsupported action type"
    @staticmethod
    def get_valid_commands() -> Dict[str, Tuple[str, Dict]]:
        """Return a dictionary of valid commands and their metadata"""
        return {k: v for k, v in Executor().VALID_COMMANDS.items()}
if __name__ == "__main__":
    executor = Executor(PROJECT_ROOT)
    executor.main()