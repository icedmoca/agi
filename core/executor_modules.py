# core/executor_modules.py

import subprocess
import shlex
from pathlib import Path
from typing import Optional

class ExecutorsModule:
    """Collection of modular executor actions"""

    @staticmethod
    def run_shell_command(command: str) -> str:
        """Run a shell command and return the output or error"""
        try:
            result = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=20
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"[stderr] {result.stderr.strip()}"
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
