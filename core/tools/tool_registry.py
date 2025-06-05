from typing import Dict, Any, Callable, Optional, List, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime
from functools import wraps
try:
    from jsonschema import validate, ValidationError  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    def validate(instance, schema):
        return True
    class ValidationError(Exception):
        pass
import shlex
import subprocess
from core.intent_classifier import classify_intent
from core.config import ALLOWED_TOOLS

logger = logging.getLogger(__name__)

def log_tool_call(func):
    """Decorator to log tool execution details"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            status = "success"
        except Exception as e:
            result = str(e)
            status = "failed"
            raise
        finally:
            trace = {
                "timestamp": datetime.now().isoformat(),
                "tool": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs),
                "status": status,
                "result": str(result),
                "duration": (datetime.now() - start_time).total_seconds()
            }
            _log_trace(trace)
        return result
    return wrapper

def _log_trace(trace: Dict[str, Any]):
    """Log execution trace to file"""
    try:
        trace_file = Path("output/traces.jsonl")
        trace_file.parent.mkdir(exist_ok=True)
        with trace_file.open("a") as f:
            f.write(json.dumps(trace) + "\n")
    except Exception as e:
        logger.error(f"Failed to log trace: {e}")

def _log_tool_trace(trace: Dict[str, Any]):
    """Log tool executions to output/tool_traces.jsonl"""
    try:
        trace_file = Path("output/tool_traces.jsonl")
        trace_file.parent.mkdir(exist_ok=True)
        with trace_file.open("a") as f:
            f.write(json.dumps(trace) + "\n")
    except Exception as e:
        logger.error(f"Failed to log tool trace: {e}")

def _normalize_result(r):
    """
    Ensure every tool result is a dict with at least 'status' and 'output'.
    """
    if isinstance(r, dict):
        # already OK
        return r
    if isinstance(r, (bytes, bytearray)):
        r = r.decode(errors="replace")
    # if we arrive here we have a str
    return {"status": "success", "output": r}

def risk_score(name: str, args: dict) -> str:
    high_risk = {"run_shell", "git_push", "evolve_file"}
    if name in high_risk:
        return "high"
    if name in {"git_commit", "heal_system"}:
        return "medium"
    return "low"

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}
        
    def register(self, name: str, func: Callable, schema: Optional[Dict[str, Any]] = None):
        """Register a tool with optional JSON schema"""
        self.tools[name] = log_tool_call(func)
        if schema:
            self.schemas[name] = schema
            
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get registered tool by name"""
        return self.tools.get(name)
        
    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get JSON schema for tool"""
        return self.schemas.get(name)
        
    def describe_tools(self) -> str:
        """Get human-readable description of all tools"""
        descriptions = []
        for name, schema in self.schemas.items():
            desc = f"Tool: {name}\n"
            if schema.get("description"):
                desc += f"Description: {schema['description']}\n"
            if schema.get("parameters"):
                desc += "Parameters:\n"
                for param, details in schema["parameters"]["properties"].items():
                    desc += f"  - {param}: {details.get('description', '')}\n"
            descriptions.append(desc)
        return "\n".join(descriptions)
        
    def run_tool(self, name: str, args: dict, max_retries: int = 0):
        """Execute a registered tool with retry and logging."""
        import time
        from core.memory import Memory
        from core.reward import score_result

        # Safety-gate: refuse execution of unlisted tools
        if name not in ALLOWED_TOOLS:
            return {"status": "error", "output": f"Tool '{name}' not allowed by policy"}

        tool_fn = self.tools[name]
        delay = 0.5
        risk = risk_score(name, args)

        # backup on high-risk actions
        if risk == "high":
            _log_tool_trace({"timestamp": datetime.now().isoformat(), "tool": name, "risk": "high", "stage": "pre-exec"})

        for attempt in range(max_retries + 1):
            start = datetime.now()
            try:
                raw = tool_fn(**args)
                result = _normalize_result(raw)
                status = result.get("status", "success")
                return result
            except Exception as e:
                result = {"status": "error", "output": str(e)}
                status = "error"
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                else:
                    return result
            finally:
                score = score_result(str(result.get("output")))
                if risk == "high":
                    # after success mark risk satisfied
                    _log_tool_trace({"timestamp": datetime.now().isoformat(), "tool": name, "risk": "high", "stage": "post-exec"})
                trace = {
                    "timestamp": datetime.now().isoformat(),
                    "tool": name,
                    "args": args,
                    "status": status,
                    "attempt": attempt,
                    "duration": (datetime.now() - start).total_seconds(),
                    "output": result.get("output"),
                    "score": score,
                }
                _log_tool_trace(trace)
                mem = Memory.latest()
                if mem:
                    meta = {
                        "type": "tool_use",
                        "args": args,
                        "status": status,
                        "task_id": args.get("task_id"),
                        "tags": args.get("tags"),
                        "file_target": args.get("file_target") or args.get("file_path") or args.get("path"),
                    }
                    mem.append(
                        goal=f"tool:{name}",
                        result=str(result.get("output")),
                        score=score,
                        metadata=meta,
                    )

# Initialize global registry
registry = ToolRegistry()

# Register core tools
@log_tool_call
def evolve_file(goal: str, file_path: str) -> str:
    """Evolve a code file based on goal"""
    from core.evolver import evolve_file as _evolve
    return _evolve(goal=goal, file_path=file_path)

@log_tool_call
def run_shell(command: str, timeout: int = 60, **kwargs) -> dict:
    """
    Execute a shell command safely and always return a dictionary.
    """
    try:
        from core.executor import Executor, is_valid_shell_command
        exec_ = Executor()

        kwargs.pop('timeout', None)
        outputs = []
        executed = 0
        skipped = 0

        for line in command.splitlines():
            if not line.strip():
                continue
            classification = classify_intent(line)
            ctype = classification.get("type", "other")
            cleaned = classification.get("value", line).strip()
            if ctype != "command":
                logger.warning("Skipped non-command: %s - %s", ctype, cleaned)
                skipped += 1
                continue
            if not is_valid_shell_command(cleaned):
                logger.warning("Skipped invalid shell command: %s", cleaned)
                skipped += 1
                continue

            proc = exec_.run_and_capture(cleaned, timeout=timeout)
            if not isinstance(proc, subprocess.CompletedProcess):
                return {"status": "error", "output": str(proc)}
            if proc.returncode != 0:
                return {
                    "status": "error",
                    "output": proc.stderr or proc.stdout,
                    "returncode": proc.returncode,
                }
            executed += 1
            outputs.append(proc.stdout)

        logger.debug("Executed %d commands, skipped %d", executed, skipped)

        combined = "\n".join(filter(None, outputs))
        return {
            "status": "success",
            "output": combined or "[SUCCESS] No output",
            "returncode": 0,
        }
            
    except Exception as e:
        return {"status": "error", "output": str(e)}

@log_tool_call
def heal_system(alert_type: str, details: Dict[str, Any]) -> Tuple[str, str]:
    """Attempt to heal system issues"""
    from core.self_heal import SelfHealer
    healer = SelfHealer()
    return healer.attempt_resolution({
        "type": alert_type,
        **details
    })

@log_tool_call
def web_search(query: str) -> str:
    """Perform a web search and return summary"""
    from core.tools.fetcher import DataFetcher
    return DataFetcher().web_search(query)

@log_tool_call
def git_commit(message: str) -> str:
    """Commit changes with a message"""
    from core.utils import commit_changes
    return commit_changes(message)

@log_tool_call
def open_browser(url: str) -> str:
    """Open a URL in the default browser"""
    import webbrowser
    webbrowser.open(url)
    return f"Opened {url}"

@log_tool_call
def file_read(path: str) -> dict:
    """Return the contents of a text file"""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return {"status": "error", "output": f"{path} not found"}
    try:
        return {"status": "success", "output": p.read_text()}
    except Exception as e:
        return {"status": "error", "output": str(e)}

@log_tool_call
def internet_fetch(url: str) -> dict:
    """Fetch raw data from a URL"""
    try:
        import requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return {"status": "success", "output": resp.text}
    except Exception:
        try:
            from urllib.request import urlopen
            with urlopen(url) as resp:
                return {"status": "success", "output": resp.read().decode()}
        except Exception as e:
            return {"status": "error", "output": str(e)}

@log_tool_call
def fetch_url(url: str) -> dict:
    """Alias for internet_fetch for compatibility"""
    return internet_fetch(url)

@log_tool_call
def read_file(path: str) -> dict:
    """Alias for file_read for compatibility"""
    return file_read(path)

@log_tool_call
def get_system_metrics() -> dict:
    """Alias for os_metrics"""
    return os_metrics()

@log_tool_call
def os_metrics() -> dict:
    """Return basic OS telemetry"""
    from core.tools.fetcher import SystemFetcher
    return SystemFetcher().get_system_telemetry()

@log_tool_call
def repo_scan(pattern: str = "*") -> dict:
    """List repository files matching a glob"""
    from pathlib import Path
    matches = [str(p) for p in Path('.').rglob(pattern)]
    return {"status": "success", "output": "\n".join(matches)}

@log_tool_call
def reflect_self() -> dict:
    """Summarise recent memory and suggest improvements"""
    from core.memory import Memory
    mem = Memory.latest()
    if not mem:
        return {"status": "error", "output": "memory unavailable"}
    recent = mem.get_recent(10)
    successes = [e for e in recent if e.score > 0][-5:]
    failures = [e for e in recent if e.score <= 0][-5:]
    summary_lines = ["Architecture modules: " + ", ".join(sorted(p.stem for p in Path('core').glob('*.py')))]
    from core.analysis import failure_stats
    stats = failure_stats(mem)
    if successes:
        summary_lines.append("Last successes:" )
        summary_lines.extend(f"- {e.goal} ({e.score})" for e in successes)
    if failures:
        summary_lines.append("Recent failures:" )
        summary_lines.extend(f"- {e.goal} ({e.score})" for e in failures)
    if stats["tags"]:
        summary_lines.append("Weak tags:" )
        summary_lines.extend(f"- {tag}: {cnt}" for tag, cnt in stats["tags"])
    suggestion = "Improve error handling" if failures else "Continue current plan"
    summary_lines.append(f"Next evolution suggestion: {suggestion}")
    return {"status": "success", "output": "\n".join(summary_lines)}

@log_tool_call
def agent_identity() -> dict:
    """Return a short self-description using memory and docs."""
    from core.memory import Memory
    mem = Memory.latest()
    successes = sum(1 for e in mem.entries if e.score > 0) if mem else 0
    failures = sum(1 for e in mem.entries if e.score <= 0) if mem else 0
    arch_path = Path("docs/ARCHITECTURE.md")
    arch_summary = arch_path.read_text().splitlines()[0] if arch_path.exists() else ""
    summary = (
        f"Who am I? An adaptive agent.\n"
        f"What do I do well? {successes} successes recorded.\n"
        f"What have I failed at? {failures} tasks.\n"
        f"{arch_summary}"
    )
    return {"status": "success", "output": summary}

@log_tool_call
def build_memory_map_tool() -> dict:
    """Generate memory map JSON grouped by tag and score."""
    from core.memory import Memory
    from core.analysis import build_memory_map
    mem = Memory.latest()
    if not mem:
        return {"status": "error", "output": "memory unavailable"}
    summary = build_memory_map(mem)
    return {"status": "success", "output": json.dumps(summary)}

# Register tools with schemas
registry.register("evolve_file", evolve_file, {
    "description": "Evolve a code file based on improvement goal",
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "Evolution goal"},
            "file_path": {"type": "string", "description": "Path to file"}
        },
        "required": ["goal", "file_path"]
    }
})

registry.register("run_shell", run_shell, {
    "description": "Execute shell command with safety checks",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command"},
            "timeout": {"type": "integer", "description": "Timeout in seconds"}
        },
        "required": ["command"]
    }
})

registry.register("heal_system", heal_system, {
    "description": "Attempt automatic system healing",
    "parameters": {
        "type": "object",
        "properties": {
            "alert_type": {"type": "string", "description": "Type of alert"},
            "details": {"type": "object", "description": "Alert details"}
        },
        "required": ["alert_type", "details"]
    }
})

registry.register("web_search", web_search, {
    "description": "Search the web for information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
})

registry.register("git_commit", git_commit, {
    "description": "Commit code changes to Git",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Commit message"}
        },
        "required": ["message"]
    }
})

registry.register("open_browser", open_browser, {
    "description": "Open a URL in the system browser",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to open"}
        },
        "required": ["url"]
    }
})

registry.register("file_read", file_read, {
    "description": "Read a text file",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"}
        },
        "required": ["path"]
    }
})

registry.register("internet_fetch", internet_fetch, {
    "description": "Fetch content from a URL",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL"}
        },
        "required": ["url"]
    }
})

registry.register("fetch_url", fetch_url, {
    "description": "Fetch content from a URL",
    "parameters": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "URL"}},
        "required": ["url"]
    }
})

registry.register("read_file", read_file, {
    "description": "Read a text file",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path"}},
        "required": ["path"]
    }
})

registry.register("get_system_metrics", get_system_metrics, {
    "description": "Retrieve basic OS metrics",
    "parameters": {"type": "object", "properties": {}}
})

registry.register("os_metrics", os_metrics, {
    "description": "Retrieve basic OS metrics",
    "parameters": {"type": "object", "properties": {}}
})

registry.register("repo_scan", repo_scan, {
    "description": "List repository files matching a glob",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern"}
        }
    }
})

registry.register("reflect_self", reflect_self, {
    "description": "Summarise recent memory and suggest self-improvements",
    "parameters": {"type": "object", "properties": {}}
})

registry.register("agent_identity", agent_identity, {
    "description": "Describe the agent using memory and docs",
    "parameters": {"type": "object", "properties": {}}
})

registry.register("build_memory_map", build_memory_map_tool, {
    "description": "Generate memory cluster map JSON",
    "parameters": {"type": "object", "properties": {}}
})

# ------------------------------------------------------------------ #
# Helper: auto-load plugins from *core/tools/plugins*
# ------------------------------------------------------------------ #

def _autoload_plugins() -> None:
    """Dynamically import *.py files from core/tools/plugins and register
    any callables tagged with `is_tool = True` attribute.
    """
    from importlib import import_module, util
    from pathlib import Path

    plugins_dir = Path(__file__).parent / "plugins"
    if not plugins_dir.exists():
        return

    for path in plugins_dir.glob("*.py"):
        mod_name = f"core.tools.plugins.{path.stem}"
        try:
            spec = util.spec_from_file_location(mod_name, path)
            if spec and spec.loader:
                module = util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore
            else:
                continue

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr) and getattr(attr, "is_tool", False):
                    tool_name = getattr(attr, "tool_name", attr.__name__)
                    registry.register(tool_name, attr, getattr(attr, "schema", None))
        except Exception as e:
            logger.error("Failed to autoload plugin %s: %s", path.name, e)


# Trigger plugin discovery at import-time
_autoload_plugins()

# ------------------------------------------------------------------ #
# Git push – new helper
# ------------------------------------------------------------------ #

@log_tool_call
def git_push(remote: str = "origin", branch: str = "HEAD") -> str:
    """Push committed changes to remote git repository."""
    from core.utils import push_changes
    return push_changes(remote, branch)

# ------------------------------------------------------------------ #
# Browse page – powered by new BrowserAgent
# ------------------------------------------------------------------ #

@log_tool_call
def browse_page(url: str) -> dict:
    """Fetch page and return plain-text content (best-effort)."""
    from core.agents.browser_agent import BrowserAgent
    agent = BrowserAgent()
    return {"status": "success", "output": agent.browse_page(url)}

registry.register("git_push", git_push, {
    "description": "Push committed code to remote repository",
    "parameters": {
        "type": "object",
        "properties": {
            "remote": {"type": "string", "description": "Remote name", "default": "origin"},
            "branch": {"type": "string", "description": "Branch/ref to push", "default": "HEAD"}
        },
        "required": []
    }
})

registry.register("browse_page", browse_page, {
    "description": "Retrieve and return the textual content of a web page",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"}
        },
        "required": ["url"]
    }
})

# ------------------------------------------------------------------ #
# External goal submission helper
# ------------------------------------------------------------------ #

@log_tool_call
def submit_external_goal(goal: str) -> dict:
    """Append a user-provided goal into *pending_goals.jsonl*."""
    from pathlib import Path
    import json, datetime

    try:
        entry = {
            "goal": goal,
            "origin": "external",
            "created": datetime.datetime.now().isoformat(),
        }
        path = Path("pending_goals.jsonl")
        with path.open("a") as fp:
            fp.write(json.dumps(entry) + "\n")
        return {"status": "success", "output": "Goal submitted"}
    except Exception as e:
        return {"status": "error", "output": str(e)}

registry.register("submit_external_goal", submit_external_goal, {
    "description": "Submit an external goal to the agent queue",
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "Goal text"}
        },
        "required": ["goal"]
    }
})

# ------------------------------------------------------------------ #
# Self identity tool
# ------------------------------------------------------------------ #

@log_tool_call
def describe_self() -> str:
    from core.identity import describe_self
    return describe_self()

registry.register("describe_self", describe_self, {
    "description": "Return markdown description of the AGI itself",
    "parameters": {"type": "object", "properties": {}},
})
