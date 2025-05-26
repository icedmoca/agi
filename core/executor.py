import subprocess
import ollama
import os
import re
from pathlib import Path
from shlex import quote
from datetime import datetime
import json
import time

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
    # Skip validation for echo commands
    if cmd.startswith("echo"):
        return True, ""
        
    # Extract paths from command
    paths = re.findall(r'(?<= )/\S+|(?<= )\./\S+', cmd)
    
    for path in paths:
        # Skip output redirection
        if path.startswith('>'):
            continue
            
        # Skip system paths
        if path.startswith(('/bin/', '/usr/bin/')):
            continue
            
        # Resolve path
        full_path = os.path.join("/home/astro/agi", path.lstrip("./"))
        
        # Check if path exists
        if not os.path.exists(full_path):
            return False, f"Path not found: {path}"
            
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
    # Use fixed project root
    PROJECT_ROOT = "/home/astro/agi"
    
    # Normalize paths
    cmd = cmd.replace("/output/", "/")
    cmd = cmd.replace("./output/", "./")
    cmd = cmd.replace("../", "./")  # Flatten relative paths
    cmd = cmd.replace("./backedup/", f"{PROJECT_ROOT}/backedup/")
    cmd = cmd.replace("./core/", f"{PROJECT_ROOT}/core/")
    
    # Fix common path mistakes
    cmd = cmd.replace("python3 /home/astro/agi/", "python3 ")
    cmd = cmd.replace("python3 ./", "python3 ")
    
    # Block unsafe/nonexistent commands
    if any(pattern in cmd.lower() for pattern in [
        "mail", "sendmail", "postfix",  # Mail commands
        "sudo", "su -", "passwd",       # Privilege escalation
        "> /dev/null", "2>&1",         # Output redirection
        "&", "nohup",                   # Background execution
        "make test", "make all",        # No Makefile
        "fail", "error",                # Invalid commands
    ]):
        return "echo '[SKIPPED] Command blocked by security policy'"
        
    # Fix path references
    cmd = cmd.replace("./", f"{PROJECT_ROOT}/")
    
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

def execute_action(goal):
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

    # Regular command handling with retries
    for attempt in range(3):
        try:
            # Build prompt with learned patterns
            prompt = f"""You are a precise shell command generator.

Goal: {goal}

Project Context:
- Working directory: {PROJECT_ROOT}
- Core modules in: ./core/
- Available files: {', '.join(os.listdir('.')[:10])}

Known issues to avoid:
{chr(10).join(f'- {p}' for p in bad_patterns)}

Generate ONE shell command to achieve the goal.
- Use absolute paths starting with {PROJECT_ROOT}
- NO placeholder paths or ../
- NO markdown formatting
- Return ONLY the raw command"""

            response = ollama.chat(
                model="mistral-hacker",
                messages=[{"role": "user", "content": prompt}],
                options={"timeout": 30}
            )
            
            cmd = response["message"]["content"].strip().split("\n")[0]
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
                timeout=60,
                text=True,
                cwd=PROJECT_ROOT  # Always execute from project root
            )
            
            # Format output
            output = []
            if proc.stdout:
                output.append(proc.stdout.strip())
            if proc.stderr:
                output.append(f"[stderr] {proc.stderr.strip()}")
                
            badge = "✅" if proc.returncode == 0 and not proc.stderr else "⚠️" if proc.stderr else "❌"
            result = f"{badge} $ {cmd}\n" + "\n".join(output) if output else f"{badge} $ {cmd}\n[No output]"
            
            return result
            
        except Exception as e:
            if attempt < 2:  # Retry on error
                time.sleep(2)
                continue
            return f"[ERROR] Failed to execute: {str(e)}"

def handle_sudo_command(cmd: str) -> str:
    """Prepare command with sudo password if needed."""
    if 'sudo' in cmd:
        # Escape the command properly for bash -c
        escaped_cmd = quote(cmd)
        # Add password input and wrap in bash -c
        return f'echo "euclid" | sudo -S bash -c {escaped_cmd}'
    return cmd

def execute_action(goal):
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

    # Regular command handling continues as before...
    try:
        prompt = f"""You are a precise shell command generator.

Goal: {goal}

Project Context:
- Working directory: {os.getcwd()}
- Project root: /home/astro/agi
- Core modules in: ./core/
- Available files: {', '.join(os.listdir('.')[:10])}

Generate ONE shell command to achieve the goal above.
- Use actual paths that exist in this project
- DO NOT use placeholder paths like /path/to/project
- DO NOT use markdown formatting or backticks
- Return ONLY the command as raw shell code."""

        response = ollama.chat(
            model="mistral-hacker",
            messages=[{"role": "user", "content": prompt}],
            options={"timeout": 30}
        )
        
        # Extract and clean command
        cmd = response["message"]["content"].strip().split("\n")[0]
        cmd = sanitize_shell_command(cmd)
        
        # Validate command
        is_valid, error_msg = validate_command(cmd)
        if not is_valid:
            return f"[INVALID COMMAND] {error_msg}"
            
        # Sanitize paths and handle sudo
        cmd = sanitize_paths(cmd)
        cmd = handle_sudo_command(cmd)
            
        print(f"[CLEANED COMMAND] {cmd}")
        
        # Create output directory if needed
        os.makedirs("./output", exist_ok=True)
        
        # Use Popen for better output handling
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="./output"  # Run in output directory
        )
        
        try:
            stdout, stderr = proc.communicate(timeout=60)  # Increased timeout for heavy commands
            
            # Log stderr for inspection
            if stderr:
                with open("./output/last_error.txt", "w") as f:
                    f.write(f"Command: {cmd}\nError: {stderr}")
            
            # Handle hash diff sync if needed
            if "diff hashes.txt trusted_hashes.txt" in cmd:
                try:
                    os.makedirs("./backedup", exist_ok=True)
                    results_path = Path("./results.txt")
                    
                    with open("./output/hash_sync.log", "a") as log:
                        log.write(f"\n=== Hash sync at {datetime.now().isoformat()} ===\n")
                        
                        if results_path.exists() and results_path.stat().st_size > 0:
                            with open(results_path, "r") as f:
                                diff_lines = f.readlines()

                            for line in diff_lines:
                                if line.startswith("< "):  # Extract changed file path
                                    parts = line.strip().split()
                                    if len(parts) >= 2:
                                        file_path = parts[1]
                                        backup_path = f"./backedup/{file_path.replace('/', '_')}"
                                        try:
                                            # Copy changed file to backup dir
                                            subprocess.run(["cp", file_path, backup_path], check=True)
                                            log.write(f"Backed up {file_path} to {backup_path}\n")

                                            # Update its hash in trusted_hashes.txt
                                            with open("trusted_hashes.txt", "a") as trust_out:
                                                subprocess.run(["sha256sum", file_path], stdout=trust_out, check=True)
                                            log.write(f"Updated hash for {file_path}\n")
                                        except Exception as e:
                                            log.write(f"[WARN] Failed to process {file_path}: {e}\n")
                except Exception as e:
                    print(f"[WARN] Hash sync failed: {e}")
                    with open("./output/hash_sync.log", "a") as log:
                        log.write(f"[ERROR] Hash sync failed: {e}\n")

            # Format output
            output = []
            if stdout:
                output.append(stdout.strip())
            if stderr:
                output.append(f"[stderr] {stderr.strip()}")
                
            if not output:
                return f"$ {cmd}\n[No output]"
                
            return f"$ {cmd}\n" + "\n".join(output)
            
        except subprocess.TimeoutExpired:
            proc.kill()
            return f"$ {cmd}\n[ERROR] Command timed out after 60 seconds"
            
    except ValueError as e:
        return f"[SYNTAX ERROR] {str(e)}"
    except Exception as e:
        return f"[ERROR] Failed to execute command: {str(e)}" 