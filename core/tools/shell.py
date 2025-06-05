def run_shell(command: str) -> dict:
    """Execute a shell command and return standardized output"""
    try:
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