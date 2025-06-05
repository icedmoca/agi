"""
Centralised run-time constants.

(Only a handful are used by core.agent_loop; extend as needed.)
"""

# seconds between main-loop iterations
SLEEP_INTERVAL: int = 30

# how many times the agent retries a failing task before giving up
MAX_RETRIES: int = 3

# ------------------------------------------------------------------ #
# Runtime safety – list of permitted tools.  Extend or restrict as
# needed for the deployment environment.  The *tool_registry* consults
# this list before executing any tool.
# ------------------------------------------------------------------ #

ALLOWED_TOOLS = [
    "evolve_file",
    "run_shell",
    "heal_system",
    "web_search",
    "git_commit",
    "git_push",
    "open_browser",
    "file_read",
    "internet_fetch",
    "fetch_url",
    "read_file",
    "get_system_metrics",
    "os_metrics",
    "repo_scan",
    "reflect_self",
    "agent_identity",
    "build_memory_map_tool",
    "browse_page",
    "submit_external_goal",
    "describe_self",
]

__all__ = ["SLEEP_INTERVAL", "MAX_RETRIES", "ALLOWED_TOOLS"] 