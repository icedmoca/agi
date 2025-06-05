"""
Centralised run-time constants.

(Only a handful are used by core.agent_loop; extend as needed.)
"""

# seconds between main-loop iterations
SLEEP_INTERVAL: int = 30

# how many times the agent retries a failing task before giving up
MAX_RETRIES: int = 3

__all__ = ["SLEEP_INTERVAL", "MAX_RETRIES"] 