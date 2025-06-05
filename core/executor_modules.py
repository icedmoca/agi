# core/executor_modules.py

import subprocess
from core.executor import is_valid_shell_command
from core.intent_classifier import classify_intent
from pathlib import Path
from typing import Optional

class ExecutorsModule:
    """Collection of modular executor actions"""

    @staticmethod
    def run_shell_command(command: str) -> str:
        """Run a shell command and return the output or error"""
        try:
            outputs = []
            executed = 0
            skipped = 0
            for line in command.splitlines():
                if not line.strip():
                    continue
                cls = classify_intent(line)
                ctype = cls.get("type", "other")
                cleaned = cls.get("value", line).strip()
                if ctype != "command":
                    skipped += 1
                    continue
                if not is_valid_shell_command(cleaned):
                    skipped += 1
                    continue
                result = subprocess.run(
                    cleaned,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=20
                )
                executed += 1
                if result.returncode != 0:
                    return f"[stderr] {result.stderr.strip()}"
                outputs.append(result.stdout.strip())
            if executed == 0:
                return "[SKIPPED]"
            return "\n".join(filter(None, outputs))
        except Exception as e:
            return f"[ERROR] {str(e)}"

    @staticmethod
    def read_file(file_path: str) -> str:
        """Safely read file content"""
        try:
            path = Path(file_path)
            if not path.exists():
                return "[ERROR] File not found"
            return path.read_text()
        except Exception as e:
            return f"[ERROR] {str(e)}"

    @staticmethod
    def write_file(file_path: str, content: str) -> str:
        """Safely write content to file"""
        try:
            Path(file_path).write_text(content)
            return "[SUCCESS] File written"
        except Exception as e:
            return f"[ERROR] {str(e)}"

    @staticmethod
    def list_directory(path: str = ".") -> str:
        """List contents of a directory"""
        try:
            entries = [str(p) for p in Path(path).iterdir()]
            return "\n".join(entries)
        except Exception as e:
            return f"[ERROR] {str(e)}"

