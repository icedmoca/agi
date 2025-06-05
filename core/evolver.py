import random
try:
    import ollama  # pragma: no cover
except Exception:  # pragma: no cover
    ollama = None
import json
from datetime import datetime
from pathlib import Path
import os
import time
import subprocess
from typing import Dict, Optional, Union, Tuple, List, Any
import logging
import warnings
import shutil
import difflib
import ast
import re
from core.vector_memory import VectorMemory
from core.reward import score_evolution
from core.reflection import ReflectionAgent
from core.memory import Memory
from core.models import MemoryEntry, Task
from core.chat import chat_with_llm, score_llm_output
from core.tools.evolve_file import evolve_file as _tools_evolve_file

# Constants
PROJECT_ROOT = "/home/astro/agi"
EVOLUTION_LOG = f"{PROJECT_ROOT}/output/evolution_log.jsonl"
CACHE_TIMEOUT = 300  # 5 minutes
MAX_RETRIES = 3
SAFE_MODE = False  # Toggle for safety checks

logger = logging.getLogger(__name__)

class Evolver:
    def __init__(self, memory_file: str = "memory.jsonl"):
        self.memory = Memory(memory_file)
        self.failed_attempts_dir = Path("output/failed_attempts")
        self.failed_attempts_dir.mkdir(parents=True, exist_ok=True)
        
    def evolve_file(self, filepath: str, goal: str, memory=None) -> dict:
        """Evolve a single file based on a goal"""
        logger.info(f"🔄 Evolving {filepath} for goal: {goal}")
        
        try:
            # Load current file content
            current_code = Path(filepath).read_text()
            
            # Get memory context
            memory_context = self._get_memory_context(filepath, goal)
            
            # Get related files being evolved together
            all_files = self._get_related_files(filepath)
            
            # Attempt evolution
            new_code = self._evolve_single_file(
                goal=goal,
                file_path=filepath,
                current_code=current_code,
                memory_context=memory_context,
                all_files=all_files
            )
            
            if not new_code:
                return {
                    "status": "failed",
                    "output": "Evolution produced no valid output"
                }
                
            # Write evolved code
            Path(filepath).write_text(new_code)
            
            return {
                "status": "completed",
                "output": "Evolution successful"
            }
            
        except Exception as e:
            error_msg = f"Evolution failed: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "failed",
                "output": error_msg
            }
            
    def _get_memory_context(self,
                            file_path: str,
                            goal: str,
                            k: int = 3) -> str:
        """
        Build a short textual context for the LLM from relevant memory
        """
        similar: List[MemoryEntry] = self.memory.find_similar(goal, top_k=k)
        if not similar:
            return ""

        lines = [f"- {e.goal}  ➜  {e.result}" for e in similar]
        return "\n".join(lines)
        
    def _get_related_files(self, filepath: str) -> List[str]:
        """Get list of files being evolved together"""
        # For now just return the single file
        # Could be expanded to include imports/dependencies
        return [filepath]
        
    def _evolve_single_file(self, goal: str, file_path: str, 
                           current_code: str, memory_context: str,
                           all_files: List[str]) -> Optional[str]:
        """Core evolution logic for a single file"""
        max_attempts = 3
        temperature = 0.7
        
        for attempt in range(max_attempts):
            logger.info(f"Evolution attempt {attempt + 1}/{max_attempts}")
            
            # Adjust temperature based on attempts
            temperature = min(0.7 + (attempt * 0.1), 0.9)
            
            # Generate evolved code
            new_code = self._generate_evolved_code(
                goal=goal,
                file_path=file_path,
                current_code=current_code,
                memory_context=memory_context,
                all_files=all_files,
                temperature=temperature
            )
            
            if new_code and self._validate_evolution(new_code):
                return new_code
                
        return None
        
    def _generate_evolved_code(self, goal: str, file_path: str,
                             current_code: str, memory_context: str,
                             all_files: List[str], temperature: float) -> Optional[str]:
        """Generate evolved code using LLM"""
        system_prompt = """You are an expert Python developer.
        Your task is to evolve Python code to meet specific goals while maintaining functionality.
        Follow all output rules exactly."""
        
        user_prompt = f"""Evolve this file to achieve the goal:

        {goal}

        STRICT OUTPUT RULES:
        - Return ONLY pure Python code
        - NO markdown, ticks, or code blocks
        - NO explanations or summaries
        - NO emojis or special characters
        - NO extra whitespace at start/end
        - Comments only if functionally necessary

        Current file ({file_path}):
        ====================
        {current_code}
        ====================

        Related files:
        {chr(10).join(f"• {Path(f).name}" for f in all_files)}

        Memory context:
        {memory_context}
        """
        
        try:
            response = chat_with_llm(
                system_prompt=system_prompt,
                user_input=user_prompt,
                format="code"
            )
            
            # Score and validate the generated code
            score = score_llm_output(response, format="code")
            logger.info(f"Generated code score: {score:.2f} for {file_path}")
            
            if score < 0.4:
                logger.warning(f"Low confidence code generation (score: {score:.2f}) for {file_path}")
                stricter_prompt = user_prompt + "\nYou MUST output ONLY valid Python code, no explanations or formatting."
                response = chat_with_llm(
                    system_prompt=system_prompt,
                    user_input=stricter_prompt,
                    format="code"
                )
                new_score = score_llm_output(response, format="code")
                logger.info(f"Retry code generation score: {new_score:.2f} for {file_path}")
                
                if new_score < 0.4:
                    logger.error(f"Code generation failed: Low confidence after retry for {file_path}")
                    return None
                
            # Clean and validate the code
            cleaned_code = response.strip()
            if not cleaned_code:
                logger.error(f"Empty code generated for {file_path}")
                return None
            
            # Validate syntax
            try:
                ast.parse(cleaned_code)
            except SyntaxError as e:
                logger.error(f"Syntax error in generated code for {file_path}: {e}")
                return None
            
            # Log successful generation
            logger.info(
                f"Code generation successful:\n"
                f"File: {file_path}\n"
                f"Score: {new_score if score < 0.4 else score:.2f}\n"
                f"Length: {len(cleaned_code)} chars"
            )
            
            return cleaned_code
            
        except Exception as e:
            logger.error(f"Code generation failed for {file_path}: {e}")
            return None
            
    def _validate_evolution(self, code: str) -> bool:
        """Validate evolved code"""
        try:
            ast.parse(code)
            return True
        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return False

def setup_logging():
    """Initialize the logging module, making it easier to track the script's execution."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Pirate speak: Log all the things!
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def is_safe_evolution(code: str) -> bool:
    """Check if the provided evolution code doesn't contain any dangerous patterns."""
    # ... (same as before)

def log_evolution(evolution: str, is_safe: bool, error: Optional[str] = None):
    """Log an evolution attempt with its success or failure and any errors that occurred during the process."""
    if not error:
        error = "No error encountered."

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'evolution': evolution,
        'is_safe': is_safe,
        'error': error,
    }

    with open(EVOLUTION_CACHE_FILE, 'w') as cachefile:
        json.dump(log_entry, cachefile)

    with open(EVOLUTION_LOG, 'a') as logfile:
        json.dump(log_entry, logfile)
        logfile.write("\n")

def load_evolution_cache():
    """Load the evolution cache from a JSON file if it exists."""
    try:
        with open(EVOLUTION_CACHE_FILE, 'r') as cachefile:
            return json.load(cachefile)
    except FileNotFoundError:
        return {}

def backup_file(file_path: str) -> str:
    """Create a timestamped backup of a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{PROJECT_ROOT}/backedup/{timestamp}_{Path(file_path).name}"
    os.makedirs(f"{PROJECT_ROOT}/backedup", exist_ok=True)
    shutil.copy(file_path, backup_path)
    print(f"📦 Backup created: {backup_path}")
    return backup_path

def confirm_diff(old: str, new: str, file_path: str):
    """Show diff and confirm changes"""
    diff = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=f"{file_path} (before)",
        tofile=f"{file_path} (after)",
        lineterm=""
    ))
    if diff:
        print("\n".join(diff))
        input("⚠️ Press [ENTER] to apply changes or Ctrl+C to abort...")

def sanitize_llm_output(raw: str) -> str:
    """Clean up LLM output to ensure only pure Python code remains"""
    # Strip code blocks (markdown, ticks, language hints)
    code = raw
    if code.startswith("```python") or code.startswith("```language=python"):
        code = code.split("```", 1)[-1]
        if code.startswith("python"):
            code = code.split("\n", 1)[-1]
    if "```" in code:
        code = code.rsplit("```", 1)[0]
    
    # Remove junk characters (emojis, etc.)
    code = re.sub(r'[^\x00-\x7F]+', '', code)

    # Remove leading whitespace that may cause "unexpected indent"
    lines = code.splitlines()
    lines = [line for line in lines if line.strip() != ""]
    min_indent = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
    normalized = "\n".join(line[min_indent:] if len(line) > min_indent else line for line in lines)

    return normalized.strip()

def strip_junk_chars(text: str) -> str:
    """Remove emojis and other non-ASCII characters from text"""
    return re.sub(r'[^\x00-\x7F]+', '', text)

def sanitize_output(text: str) -> str:
    """Remove invisible Unicode characters and control chars from text"""
    # Track original length for logging
    orig_len = len(text)
    
    # Remove BOM, zero-width space, and other invisible Unicode
    text = text.replace('\ufeff', '')  # BOM
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\r', '')      # Carriage return
    
    # Remove control chars except newline and tab
    clean = ''.join(char for char in text if char == '\n' or char == '\t' or ord(char) >= 32)
    
    # Log if any characters were removed
    if len(clean) != orig_len:
        logging.warning(f"Removed {orig_len - len(clean)} invisible/control characters from LLM output")
        
    return clean

def run_pytest(file_path: str) -> tuple[bool, str]:
    """Run pytest on a file and return success status and output"""
    try:
        result = subprocess.run(
            ["pytest", file_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout
    except Exception as e:
        return False, str(e)

def get_diff_size(old_code: str, new_code: str) -> int:
    """Calculate the size of changes between two code versions"""
    from difflib import unified_diff
    diff = list(unified_diff(
        old_code.splitlines(),
        new_code.splitlines()
    ))
    return len(diff)

def check_evolution_patterns(memory, file_path: str, goal: str) -> str:
    reflection = ReflectionAgent(memory)
    
    # Get recent evolutions for this file
    recent = memory.search(f"evolution {file_path}", top_k=5)
    
    # Count similar goals
    similar_goals = sum(1 for mem in recent if goal in mem)
    
    # Check for flapping (undo/redo)
    flap_count = 0
    for i in range(len(recent)-1):
        if recent[i].split(" ")[-1] == recent[i+1].split(" ")[-1]:
            flap_count += 1
            
    # Generate warnings
    warnings = []
    if similar_goals >= 3:
        warnings.append(f"⚠️ Goal '{goal}' has been attempted {similar_goals} times")
    if flap_count >= 2:
        warnings.append("⚠️ Detected potential undo/redo loop")
        
    return "\n".join(warnings)

def extract_imports(code: str) -> list[str]:
    """Extract import statements from Python code"""
    try:
        tree = ast.parse(code)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                for name in node.names:
                    imports.append(f"{module}.{name.name}" if module else name.name)
        return imports
    except:
        return []

def resolve_project_imports(imports: list[str], project_root: str) -> list[str]:
    """Convert import statements to actual project file paths"""
    project_files = []
    for imp in imports:
        # Convert dot notation to path
        path_parts = imp.split('.')
        potential_paths = [
            Path(project_root) / '/'.join(path_parts) / '__init__.py',
            Path(project_root) / f"{'/'.join(path_parts)}.py",
            Path(project_root) / 'core' / f"{'/'.join(path_parts)}.py",
        ]
        
        for path in potential_paths:
            if path.exists():
                project_files.append(str(path))
                break
                
    return project_files

def update_dependency_map(file_path: str, dependencies: list[str]):
    """Update the dependency map with new file relationships"""
    map_path = Path("core/dependency_map.json")
    dep_map = {}
    
    if map_path.exists():
        with open(map_path) as f:
            dep_map = json.load(f)
            
    dep_map[file_path] = list(set(dependencies))
    
    with open(map_path, 'w') as f:
        json.dump(dep_map, f, indent=2)

def evolve_files(goal: str, file_path: str, related_files: List[str]) -> List[str]:
    """Evolve multiple files based on goal"""
    results = []
    
    # Get related memory entries
    related_mem = memory.find_similar(goal, top_k=3)
    
    # Convert memory entries to strings
    related_snippets = "\n\n".join([
        f"Memory entry: {entry.get('goal', '')} - {entry.get('result', '')}"
        for entry in related_mem
    ])
    
    # Rest of the function...

def clean_generated_code(text: str) -> str:
    """Clean and sanitize generated code output"""
    import re
    
    # Remove markdown code fences
    text = re.sub(r"^```(?:python)?", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE) 
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)

    # Remove invisible/unprintable characters
    text = ''.join(c for c in text if c.isprintable())

    # Remove stray emojis or non-ASCII
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Drop bad lines like shebangs or encoding headers
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        if re.match(r'^#!', line) or 'coding' in line:
            continue
        clean_lines.append(line)
        
    return '\n'.join(clean_lines).strip()

def validate_evolution_goal(goal: str, content: str) -> bool:
    """Validate if evolution satisfies the goal requirements"""
    goal_lower = goal.lower()
    
    # Logging setup requirements
    if "add logging" in goal_lower or "fix logging" in goal_lower:
        required = [
            "import logging",
            "logger = logging.getLogger"
        ]
        return all(req in content for req in required)
    
    # Import requirements
    if "add import" in goal_lower:
        import_name = re.search(r"add import (\w+)", goal_lower)
        if import_name:
            return f"import {import_name.group(1)}" in content
            
    # Code cleanup goals
    if "remove print" in goal_lower:
        return "print(" not in content
        
    # Modern Python practices
    if "use pathlib" in goal_lower:
        return "from pathlib import" in content and "os.path" not in content
        
    # Type hints
    if "add type hints" in goal_lower or "add typing" in goal_lower:
        return "from typing import" in content
        
    return True

def evolve_single_file(goal: str, file_path: str, current_code: str, 
                      memory_context: str, all_files: list[str]) -> str:
    """Evolve a single file with awareness of related files"""
    try:
        # ... existing code ...
        
        # Get LLM response with adjusted temperature
        raw_output = chat_with_llm(
            system_prompt, 
            user_prompt,
            format="code"
        )
        
        # Score the output
        score = score_llm_output(raw_output, format="code")
        logger.info(f"Evolution output score: {score:.2f}")
        
        if score < 0.4:
            logger.warning(f"Low confidence evolution (score: {score:.2f})")
            # Retry with stricter prompt
            stricter_prompt = user_prompt + "\nYou MUST respond with valid Python code only, no explanations or markdown."
            raw_output = chat_with_llm(
                system_prompt,
                stricter_prompt,
                format="code"
            )
            new_score = score_llm_output(raw_output, format="code")
            logger.info(f"Retry evolution score: {new_score:.2f}")
            
            if new_score < 0.4:
                return "[FAILED] Low-confidence evolution output"
        
        if not raw_output:
            logger.error("❌ All models failed to generate valid output")
            return "[ERROR] No valid output generated"

        # Clean and validate content
        content = clean_generated_code(raw_output).strip()
        
        # Add debug logging
        logger.debug("⚠️ Code Preview:\n%s", content[:200])
        
        if not content:
            logger.error("❌ Output empty after cleaning")
            return "[ERROR] Empty output after cleaning"

        # AST validation
        try:
            ast.parse(content)
        except Exception as e:
            logger.error(f"❌ Syntax error: {e}")
            return f"[ERROR] Syntax validation failed: {str(e)}"

        # After AST validation:
        if not validate_evolution_goal(goal, content):
            # Log failed attempt for inspection
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_file = failed_attempts_dir / f"{timestamp}_attempt_{attempt}.py"
            failed_file.write_text(content)
            logger.warning(f"❌ Failed attempt logged to {failed_file}")
            return "[FAILED] Evolution validation failed"
            
        return content

    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        return f"[ERROR] Evolution failed: {str(e)}"

def evolve_file(goal: str, file_path: str) -> str:
    """Evolve a single file based on a goal"""
    try:
        # ... existing code ...
        
        # Generate evolved code
        response = chat_with_llm(prompt, format="code")
        
        # Score the evolved code
        code_score = score_llm_output(response, format="code")
        logger.info(f"Evolution code score: {code_score:.2f}")
        
        if code_score < 0.4:
            logger.warning(f"Low confidence evolution (score: {code_score:.2f})")
            # Retry with stricter prompt
            stricter_prompt = prompt + "\nProvide ONLY valid Python code without any explanations or markdown."
            response = chat_with_llm(stricter_prompt, format="code")
            new_score = score_llm_output(response, format="code")
            logger.info(f"Retry evolution score: {new_score:.2f}")
            
            if new_score < 0.4:
                return "[FAILED] Low-confidence evolution output"
        
        # Validate and apply changes
        if validate_code(response):
            # Apply the changes
            with open(file_path, 'w') as f:
                f.write(response)
            return "[SUCCESS] Evolution applied"
        else:
            return "[FAILED] Code validation failed"
        
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        return f"[ERROR] Evolution failed: {str(e)}"

def handle_evolution(goal: str, file_path: str) -> str:
    """Handle evolution of a single file"""
    try:
        # Get file content
        with open(file_path) as f:
            original = f.read()
            
        # Generate and apply changes
        result = evolve_file(goal, file_path)
        
        # Score the evolution result
        score = score_llm_output(result, format="code")
        logger.info(f"Evolution result score: {score:.2f} for {goal}")
        
        # Score the evolution
        with open(file_path) as f:
            new_content = f.read()
        diff_size = len(new_content) - len(original)
        evolution_score = score_evolution(goal, result, diff_size)
        
        # Log both scores
        logger.info(f"Evolution scores - LLM: {score:.2f}, Evolution: {evolution_score}")
        
        return result
        
    except Exception as e:
        return f"[ERROR] Evolution failed: {str(e)}"

# --------------------------------------------------------------------------- #
# Public helper – keeps backward-compat with older tests / scripts
# --------------------------------------------------------------------------- #
def evolve_file(task: Task, memory: Memory | None = None, **kwargs) -> str:
    """
    Thin wrapper that forwards to core.tools.evolve_file but maintains the
    historical import path and keyword signature required by tests.
    """
    return _tools_evolve_file(task, memory, **kwargs)

# ------------------------------------------------------------------ #
# LLM helper – now backed by Ollama                                 #
# ------------------------------------------------------------------ #
def chat_with_llm(
    system_prompt: Optional[str],
    user_input: str,
    memory: Optional[Memory] = None,
    *,
    model: str = "mistral-hacker",
    timeout: int = 60,
) -> str:
    """
    Send *user_input* (and optional *system_prompt*) to the local Ollama
    server and return the model's reply.  Any exception is logged and
    surfaced to the caller as an `[ERROR]` string.
    """
    # ------------------------------------------------------------------ #
    # Persistent instruction – nudges the LLM toward safe, runnable cmds #
    # ------------------------------------------------------------------ #
    default_system = (
        "You are an autonomous shell-integrated AI. "
        "Wrap shell commands in triple back-ticks with the language tag 'bash' when appropriate. "
        "Use absolute paths to binaries when possible (e.g., /usr/bin/lsof). "
        "Avoid 'sudo' unless necessary — if you must use it, assume password is 'euclid'. "
        "Short inline commands like `ls` or `uname -a` may also be returned as single lines."
    )

    # ------------------------------------------------------------------ #
    # Helper: run a shell command, stream output, and return the capture  #
    # ------------------------------------------------------------------ #
    from typing import List

    def run_and_stream(command: str) -> str:
        """
        Execute *command* with /bin/bash, streaming output to the console
        while also capturing it so we can feed the result back to the LLM.
        """
        if command.startswith("sudo "):
            command = f"echo 'euclid' | sudo -S {command[5:].strip()}"

        # curl helper – quote URLs that contain '?'
        if command.startswith("curl "):
            m_url = re.search(r"(https?://\S+)", command)
            if m_url:
                url = m_url.group(1)
                if "?" in url and '"' not in url:
                    command = command.replace(url, f'"{url}"')

        env = os.environ.copy()
        env["PATH"] = env.get("PATH", "") + ":/usr/local/bin:/usr/bin:/bin:/sbin:/usr/sbin"

        print(f"\n🔧 Running: {command}\n")
        logging.info("[chat_with_llm] live-exec: %s", command)

        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            executable="/bin/bash",
            env=env,
        )

        output_lines: List[str] = []
        if proc.stdout:
            for line in proc.stdout:
                print(line, end="", flush=True)  # real-time stream
                output_lines.append(line)
        proc.wait()
        return "".join(output_lines).strip()

    # ------------------------------------------------------------------ #
    # 1. Detect fenced bash/sh blocks in *user_input*                    #
    # 2. Execute them sequentially                                       #
    # 3. Append their combined output to the prompt                      #
    # ------------------------------------------------------------------ #
    command_blocks = re.findall(r"```(?:bash|sh)\n(.*?)```", user_input, flags=re.DOTALL)
    if command_blocks:
        outputs: List[str] = []
        for block in command_blocks:
            cmd = block.strip()
            if cmd:
                outputs.append(run_and_stream(cmd))

        combined_output = "\n".join(outputs).strip()

        if combined_output:
            print("\n📄 Command output:\n" + combined_output + "\n")

            # Feed the fresh output back to the LLM so it can chat naturally
            user_input = (
                f"{user_input}\n\n[Command Output]:\n{combined_output}\n\n"
                "Now respond like you're chatting with a human. "
                "Be friendly, brief, and natural."
            )

    # ------------------------------------------------------------------ #
    # Build the message list *after* we may have modified user_input      #
    # ------------------------------------------------------------------ #
    messages: List[Dict[str, str]] = []

    if system_prompt:
        # Prepend the default guard-rails before any custom instructions
        messages.append(
            {
                "role": "system",
                "content": f"{default_system}\n\n{system_prompt}",
            }
        )
    else:
        messages.append({"role": "system", "content": default_system})

    # Finally add the user's message
    messages.append({"role": "user", "content": user_input})

    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"timeout": timeout},
        )
        original_reply = response["message"]["content"].strip()

        # --------------------------------------------------------- #
        # 1. Collect *all* shell commands (fenced + single-line)    #
        # 2. Run them sequentially with live output streaming       #
        # --------------------------------------------------------- #

        def extract_shell_commands(text: str) -> List[str]:
            """
            Return a list of shell commands found in *text*.
            – commands inside ```bash …``` / ```sh …``` blocks
            – stand-alone lines that *look* like commands (best-effort)
            """
            # Fenced blocks
            blocks = re.findall(r"```(?:bash|sh)?\n(.*?)```", text, flags=re.DOTALL)

            # Very permissive single-line matcher
            inlines = re.findall(r"^(?:sudo\s+)?[\w./-]+(?:\s+[^`]+)?$", text, flags=re.MULTILINE)

            # De-duplicate while preserving order
            seen, cmds = set(), []
            for cmd in blocks + inlines:
                cmd = cmd.strip()
                if cmd and cmd not in seen:
                    seen.add(cmd)
                    cmds.append(cmd)
            return cmds

        commands_to_run = extract_shell_commands(original_reply)

        if commands_to_run:
            # Execute every extracted command and collect their outputs
            outputs: List[str] = [run_and_stream(cmd) for cmd in commands_to_run]
            command_output = "\n".join(outputs).strip()

            # Replace the assistant reply with a fenced block containing
            # the real-world output.  If execution produced *no* output,
            # fall back to the original model response.
            if command_output:
                reply = f"```output\n{command_output}\n```"
            else:
                reply = original_reply
        else:
            # No runnable commands detected – use the model's reply verbatim
            reply = original_reply

        # Optional long-term memory logging
        if memory:
            try:
                memory.append(
                    goal=user_input,
                    result=reply,
                    metadata={
                        "type": "llm_chat",
                        "model": model,
                        "raw_llm_reply": original_reply,
                    },
                )
            except Exception:
                # Never allow memory failure to bubble up
                pass

        return reply

    except Exception as exc:
        logging.error(f"LLM call failed: {exc}")
        reply = f"[ERROR] LLM call failed: {exc}"
        return reply


def safe_apply_evolution(file_path: str, new_code: str, goal: str, memory: Memory) -> dict:
    """Safely apply evolved code with backups and logging"""
    cleaned = sanitize_llm_output(new_code)
    try:
        ast.parse(cleaned)
    except Exception as e:
        return {"status": "error", "output": f"Invalid syntax: {e}"}

    original = Path(file_path).read_text()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = Path(f"{file_path}.{timestamp}.bak")
    shutil.copy(file_path, backup)

    Path(file_path).write_text(cleaned)

    diff = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            cleaned.splitlines(),
            fromfile="before",
            tofile="after",
        )
    )

    log_line = f"{datetime.now().isoformat()} | {file_path} | {goal} | diff_size:{len(diff.splitlines())}\n"
    Path("evolution_log.md").open("a").write(log_line)

    memory.append(
        goal=f"evolve:{file_path}",
        result="[SUCCESS] evolution applied",
        metadata={"diff": diff, "backup": str(backup)},
    )

    return {"status": "success", "output": "applied", "diff": diff}
