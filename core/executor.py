import subprocess
import ollama
import os
import re
from pathlib import Path
from shlex import quote
from datetime import datetime
import json
import time
import sys

assert "execute_action" not in globals(), "Duplicate definition of execute_action"

def sanitize_shell_command(command: str) -> str:
    """Clean up shell command by removing markdown and fixing syntax."""
    # Remove markdown code blocks
    command = re.sub(r"^```(bash|sh)?", "", command)
    command = command.replace("```", "")
    
    # Remove any remaining backticks
    command = re.sub(r'`([^`]+)`', r'\1', command)
    command = re.sub(r'\$\(([^)]+)\)', r'\1', command)
    
    # Fix common syntax issues
    command = command.strip()
    
    # Check for unclosed quotes
    if command.count('"') % 2 != 0 or command.count("'") % 2 != 0:
        raise ValueError("Unclosed quotes in command")
    
    # Fix unescaped parentheses in grep patterns
    if 'grep' in command:
        # Escape parentheses in grep patterns
        parts = command.split()
        for i, part in enumerate(parts):
            if part.startswith('-E') or part.startswith('-P'):
                # Don't escape already escaped parentheses
                if '\\(' not in part and '\\)' not in part:
                    parts[i] = part.replace('(', '\\(').replace(')', '\\)')
        command = ' '.join(parts)
    
    return command

def validate_command(cmd: str) -> tuple[bool, str]:
    """Validate command before execution"""
    PROJECT_ROOT = "/home/astro/agi"
    
    # Block multi-line commands
    if "\n" in cmd or "\\" in cmd:
        return False, "Multi-line commands not allowed"
        
    # Ensure absolute paths for Python files
    if "python3" in cmd and not cmd.startswith("python3 /"):
        return False, "Python commands must use absolute paths"
        
    # Validate directory exists for cd commands
    if cmd.startswith("cd "):
        target_dir = cmd.split("&&")[0].replace("cd ", "").strip()
        if not os.path.isdir(target_dir):
            return False, f"Directory not found: {target_dir}"
            
    return True, ""

def sanitize_paths(cmd: str) -> str:
    """Rewrite absolute paths to be relative to current directory."""
    # Replace /root/ paths with ./
    cmd = re.sub(r'>\s*/root/([^/\s]+)', r'> ./\1', cmd)
    cmd = re.sub(r'>>\s*/root/([^/\s]+)', r'>> ./\1', cmd)
    
    # Replace other absolute paths in output redirection
    cmd = re.sub(r'>\s*/(?:home|usr|var)/\w+/([^/\s]+)', r'> ./\1', cmd)
    cmd = re.sub(r'>>\s*/(?:home|usr|var)/\w+/([^/\s]+)', r'>> ./\1', cmd)
    
    return cmd

def handle_sudo_command(cmd: str) -> str:
    """Prepare command with sudo password if needed."""
    if 'sudo' in cmd:
        # Escape the command properly for bash -c
        escaped_cmd = quote(cmd)
        # Add password input and wrap in bash -c
        return f'echo "euclid" | sudo -S bash -c {escaped_cmd}'
    return cmd

def clean_command(cmd: str) -> str:
    """Sanitize commands by removing unsafe patterns and resolving paths"""
    PROJECT_ROOT = "/home/astro/agi"
    
    # Clean up multi-line commands and backslashes
    cmd = cmd.replace('\\\n', ' ').replace('\\', ' ').strip()
    
    # Block obviously invalid commands
    if "../../" in cmd or cmd.endswith('\\'):
        return "echo '[INVALID] Multi-line or invalid path detected'"
        
    # Fix Python execution paths with regex
    cmd = re.sub(r'python3\s+\.\./(\w+\.py)', f'python3 {PROJECT_ROOT}/\\1', cmd)
    cmd = re.sub(r'python3\s+\./(\w+\.py)', f'python3 {PROJECT_ROOT}/\\1', cmd)
    cmd = re.sub(r'python3\s+(\w+\.py)', f'python3 {PROJECT_ROOT}/\\1', cmd)
    
    # Handle test commands more gracefully
    if "main.py tests" in cmd:
        return f"python3 -m pytest {PROJECT_ROOT}/tests"
    if "test" in cmd and ".py" in cmd:
        return f"python3 -m py_compile {PROJECT_ROOT}/core/executor.py"
        
    # Fix directory navigation
    if cmd.startswith("cd"):
        parts = cmd.split("&&")
        if len(parts) > 1:
            # Clean up cd command and ensure it uses PROJECT_ROOT
            cd_cmd = parts[0].strip()
            if not cd_cmd.startswith("cd /"):
                cd_cmd = f"cd {PROJECT_ROOT}"
            rest = " && ".join(parts[1:]).strip()
            cmd = f"{cd_cmd} && {rest}"
        else:
            # Single cd command
            if not cmd.startswith("cd /"):
                cmd = f"cd {PROJECT_ROOT}"
                
    # Update prompt template for better command generation
    if "execute_action" in cmd:
        prompt_template = f"""You are a precise shell command generator.
Goal: {{goal}}

Project Context:
- Working directory: {PROJECT_ROOT}
- Core modules in: {PROJECT_ROOT}/core/
- Available files: {{files}}

Requirements:
- Generate ONE SINGLE LINE command (no backslashes or line breaks)
- Use absolute paths starting with {PROJECT_ROOT}
- NO placeholder paths or ../
- NO markdown formatting
- Return ONLY the raw command"""

        cmd = cmd.replace("{{PROMPT}}", prompt_template)
    
    return cmd

def validate_paths(cmd: str) -> bool:
    """Check if all files/paths in command exist"""
    # Extract paths from command
    path_patterns = [
        r'(?<= )/\S+',  # Absolute paths
        r'(?<= )\./\S+',  # Relative paths starting with ./
        r'(?<= )[^-]\w+/\S+',  # Other relative paths
    ]
    
    paths = []
    for pattern in path_patterns:
        paths.extend(re.findall(pattern, cmd))
    
    # Validate each path
    for path in paths:
        # Skip output redirection
        if path.startswith('>') or path.startswith('>>'):
            continue
            
        # Skip common commands
        if path.startswith('/bin/') or path.startswith('/usr/bin/'):
            continue
            
        # Check if path exists
        real_path = os.path.expanduser(path)
        if not os.path.exists(real_path):
            return False
            
    return True

def is_safe_command(cmd: str) -> bool:
    """Check if command is safe to execute"""
    # Banned patterns and placeholders
    unsafe_patterns = [
        "path/to", "/path/to",  # Placeholder paths
        "`", "<(", ">(", "&",   # Unsafe shell syntax
        "code ", "vim ", "emacs ", "nano ",  # GUI/Interactive editors
        "git ", "make ",  # Build tools that need review
        "./test", "./script",  # Unvalidated scripts
        "sudo", "rm -rf",  # Dangerous operations
        "|", "&&", "||"  # Complex shell operations
    ]
    
    # Check for unsafe patterns
    if any(pattern in cmd.lower() for pattern in unsafe_patterns):
        return False
        
    # Validate all paths exist
    paths = re.findall(r'(?<= )/\S+|(?<= )\./\S+', cmd)
    for path in paths:
        # Skip output redirection
        if path.startswith('>'):
            continue
        # Skip common system paths
        if path.startswith(('/bin/', '/usr/bin/')):
            continue
        # Check if path exists
        if not os.path.exists(os.path.expanduser(path)):
            return False
            
    return True

def generate_command(goal: str, bad_patterns: set) -> str:
    """Generate a command using AI with proper path handling"""
    PROJECT_ROOT = "/home/astro/agi"
    
    prompt = f"""You are a precise shell command generator for an AGI system.

Goal: {goal}

CRITICAL REQUIREMENTS:
- Generate EXACTLY ONE command on a single line
- ALL Python commands MUST start with: python3 {PROJECT_ROOT}/
- Example: python3 {PROJECT_ROOT}/core/executor.py
- NO relative paths like ../executor.py or ./executor.py
- NO multi-line commands or backslashes
- NO placeholder paths or /path/to/project

Project Context:
- Working directory: {PROJECT_ROOT}
- Core modules: {PROJECT_ROOT}/core/
- Available files: {', '.join(os.listdir('.')[:10])}

Known issues to avoid:
{chr(10).join(f'- {p}' for p in bad_patterns)}

Return ONLY the raw shell command."""

    print(f"🔍 PROMPT SENT TO AI:\n{prompt}\n")

    response = ollama.chat(
        model="mistral-hacker",
        messages=[{"role": "user", "content": prompt}],
        options={"timeout": 30}
    )
    
    return response["message"]["content"].strip().split("\n")[0]

def execute_action(goal: str) -> str:
    """Execute a shell command with improved safety and persistence"""
    PROJECT_ROOT = "/home/astro/agi"
    BAD_PATTERNS_FILE = f"{PROJECT_ROOT}/output/bad_patterns.jsonl"
    
    # Load known bad patterns
    bad_patterns = set()
    if os.path.exists(BAD_PATTERNS_FILE):
        with open(BAD_PATTERNS_FILE, 'r') as f:
            for line in f:
                try:
                    pattern = json.loads(line)['pattern']
                    bad_patterns.add(pattern)
                except:
                    continue

    # Handle specialized log analysis
    if "analyze logs" in goal.lower():
        try:
            # Create output directory and set output file
            os.makedirs("./output", exist_ok=True)
            output_file = "log_analysis.txt"
            output_path = f"./output/{output_file}"
            
            # Construct log analysis command with proper root access and file handling
            command = f"""echo 'euclid' | sudo -S bash -c '
                mkdir -p ./output &&
                find /var/log -type f \\( -name "*.log" -o -name syslog -o -name auth.log \\) -readable ! -name "{output_file}" 2>/dev/null |
                while read -r file; do
                    echo "=== Scanning $file at $(date) ===" >> "{output_path}"
                    grep -Ei "suspicious|unusual|warning|error|critical|failed|denied|attack" "$file" 2>/dev/null >> "{output_path}"
                    echo "=== End of $file ===" >> "{output_path}"
                    echo "" >> "{output_path}"
                done &&
                chown $SUDO_USER:$SUDO_USER "{output_path}"'"""
            
            # Execute the analysis
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="."
            )
            
            stdout, stderr = proc.communicate(timeout=60)
            
            if proc.returncode == 0:
                return "Log analysis completed and saved to ./output/log_analysis.txt"
            else:
                return f"[ERROR] Log analysis failed: {stderr}"
                
        except Exception as e:
            return f"[ERROR] Failed to analyze logs: {str(e)}"

    # Regular command handling with retries
    for attempt in range(3):
        try:
            cmd = generate_command(goal, bad_patterns)
            cmd = clean_command(cmd)
            
            # Validate command
            is_valid, error_msg = validate_command(cmd)
            if not is_valid:
                # Log bad pattern for future avoidance
                with open(BAD_PATTERNS_FILE, 'a') as f:
                    json.dump({"pattern": error_msg, "timestamp": datetime.now().isoformat()}, f)
                    f.write('\n')
                if attempt < 2:  # Try again with updated patterns
                    time.sleep(2)
                    continue
                return f"[INVALID] {error_msg}"
                
            # Execute with proper path handling
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            return proc.stdout if proc.returncode == 0 else f"[ERROR] {proc.stderr}"
            
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            return f"[ERROR] {str(e)}"
            
    return "[ERROR] All attempts failed"

def handle_sudo_command(cmd: str) -> str:
    """Prepare command with sudo password if needed."""
    if 'sudo' in cmd:
        # Escape the command properly for bash -c
        escaped_cmd = quote(cmd)
        # Add password input and wrap in bash -c
        return f'echo "euclid" | sudo -S bash -c {escaped_cmd}'
    return cmd

def run_tests():
    """Run test suite when module is executed directly"""
    print("Running executor tests...")
    test_cases = [
        ("cd /home/astro/agi && python3 core/executor.py", True),
        ("python3 ../executor.py tests", False),
        ("cd core && python3 executor.py", True),
        ("python3 /home/astro/agi/core/executor.py", True),
    ]
    
    for cmd, expected in test_cases:
        cleaned = clean_command(cmd)
        is_valid, _ = validate_command(cleaned)
        print(f"\nTest: {cmd}")
        print(f"Cleaned: {cleaned}")
        print(f"Valid: {is_valid} (Expected: {expected})")
        
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests() 