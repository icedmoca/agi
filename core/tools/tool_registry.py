from typing import Dict, Any, Callable, Optional, List, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime
from functools import wraps
from jsonschema import validate, ValidationError
import shlex
import subprocess

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

        tool_fn = self.tools[name]
        delay = 0.5
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
                trace = {
                    "timestamp": datetime.now().isoformat(),
                    "tool": name,
                    "args": args,
                    "status": status,
                    "attempt": attempt,
                    "duration": (datetime.now() - start).total_seconds(),
                    "output": result.get("output"),
                }
                _log_tool_trace(trace)
                mem = Memory.latest()
                if mem:
                    mem.append(
                        goal=f"tool:{name}",
                        result=str(result.get("output")),
                        metadata={"type": "tool_use", "args": args, "status": status},
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
        from core.executor import Executor
        exec_ = Executor()
        # Remove timeout from kwargs since we're passing it explicitly
        kwargs.pop('timeout', None)
        proc = exec_.run_and_capture(command.split(), timeout=timeout)
        
        # Always return a properly formatted dictionary
        if isinstance(proc, str):
            return {"status": "error", "output": proc}
        elif isinstance(proc, tuple):
            returncode, stdout, stderr = proc
            if returncode != 0:
                return {"status": "error", "output": stderr or stdout, "returncode": returncode}
            return {"status": "success", "output": stdout or "[SUCCESS] No output", "returncode": returncode}
        else:
            return {"status": "error", "output": "Invalid response format from executor"}
            
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
    import requests
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return {"status": "success", "output": resp.text}
    except Exception as e:
        return {"status": "error", "output": str(e)}

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
