python
import re
import os
import json
import time
import sys
import subprocess
from datetime import datetime
import ollama
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from logging import getLogger, StreamHandler, Formatter, ERROR, INFO
from core.executor_modules import validate_command, execute_command, execute, execute_action
from core.evolver import evolve_file
# Constants
VALID_COMMANDS = {
    "validate code": ("python3 -m py_compile", {"status": "code validation", "result": None}),
    "check syntax": ("python3 -m py_compile", {"status": "syntax check", "result": None}),
    "verify file": ("ls -l", {"status": "file verification"}),
    "list backups": ("ls /home/astro/agi/backedup/", {"status": "backup list"}),
    "run tests": ("python3 -m pytest /home/astro/agi/tests", {"status": "test execution", "result": None})
}
PROJECT_ROOT = "/home/astro/agi"
BAD_PATTERNS_FILE = f"{PROJECT_ROOT}/output/bad_patterns.jsonl"
FALLBACK_EXECUTOR = f"{PROJECT_ROOT}/core/executor.py"
# Added: logging configuration and initialization
LOGGER = getLogger(__name__)
LOG_FORMATTER = Formatter(f"[%(asctime)s] %(levelname)s: %(message)s")
LOG_STREAM_HANDLER = StreamHandler()
LOG_STREAM_HANDLER.setFormatter(LOG_FORMATTER)
LOGGER.addHandler(LOG_STREAM_HANDLER)
LOGGER.setLevel(ERROR)
def sanitize_command(cmd: str) -> str:
    """Sanitize commands by removing unsafe patterns"""
    cmd = re.sub(r'[`;&|]', '', cmd)
    return cmd.strip()
def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
def run_tests():
    execute_action("run tests")
def execute_action(goal_description: str) -> str:
    """Top-level action executor for AGI goals"""
    if goal_description.startswith("evolve "):
        try:
            # Extract file and evolution goal
            parts = goal_description.split(" to ")
            filename = parts[0].replace("evolve ", "").strip()
            improvement = parts[1].strip()
            # Validate file is in core directory
            if not filename.endswith('.py'):
                return "[ERROR] Only Python files can be evolved"
            file_path = f"{PROJECT_ROOT}/core/{filename}"
            if not os.path.isfile(file_path):
                return "[ERROR] File not found or unsafe request"
            # Call evolver with extracted parameters
            print(f" Evolution Request:\nFile: {filename}\nGoal: {improvement}")
            return evolve_file(goal=improvement, file_path=file_path)
        except Exception as e:
            return f"[ERROR] Evolution failed: {str(e)}"
    elif goal_description in VALID_COMMANDS:
        command = VALID_COMMANDS[goal_description][0]
        return execute_command(command, VALID_COMMANDS[goal_description][1])
    else:
        LOGGER.error("Unsupported action type")
        return "[ERROR] Unsupported action type"