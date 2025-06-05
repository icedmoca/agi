from core.executor import is_valid_shell_command
import logging

logger = logging.getLogger(__name__)

def run_shell(command: str) -> dict:
    """Execute a shell command and return standardized output"""
    try:
        if not is_valid_shell_command(command):
            logger.warning("Skipped invalid shell command: %s", command)
            return {"status": "error", "output": "Rejected non-shell command", "returncode": -1}
        import subprocess
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True
        )
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "output": result.stdout + result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        return {
            "status": "error",
            "output": str(e),
            "returncode": -1
        }
