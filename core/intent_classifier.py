import json
import logging
import re
from typing import Dict, Any
from enum import Enum

from .chat import chat_with_llm
from core.memory import Memory

logger = logging.getLogger(__name__)

# Patterns that typically cause the shell to wait for user input
_INTERACTIVE_PATTERNS = [
    r"\bread\b",
    r"\bselect\b",
    r"\bpause\b",
    r"\btrap\b",
    r"\binput\b",
    r"sleep\s+infinity",
    r"while\s+true",
    r"\bexpect\b",
]

_CLASSIFY_PROMPT = (
    "You are an intent classifier. Categorize the given text into one of the"
    " following types: command, explanation, instruction, error, suggestion,"
    " other. Respond ONLY with JSON like {\"type\": \"command\"}."
)

# Keep a local enum fallback to avoid circular import during type-checking
class _TaskType(str, Enum):
    CHAT = "chat"
    COMMAND = "command"
    TOOL = "tool"
    EVOLUTION = "evolution"
    REFLECTION = "reflection"
    FETCH = "fetch"
    MEMORY_QUERY = "memory_query"

def _keyword_detect(goal: str) -> _TaskType:
    g = goal.lower()
    if any(k in g for k in ("fix", "refactor", "evolve", "modify")):
        return _TaskType.EVOLUTION
    if any(k in g for k in ("list", "show", "run", "execute")) and len(g.split()) <= 7:
        return _TaskType.COMMAND
    if any(k in g for k in ("push", "commit", "search", "fetch", "download")):
        return _TaskType.TOOL
    if "weather" in g or "stock" in g or g.endswith("?"):
        return _TaskType.FETCH
    if "who are you" in g or "describe" in g:
        return _TaskType.CHAT
    return _TaskType.CHAT

def classify_intent(text: str) -> Dict[str, Any]:
    """Heuristic + memory-driven intent classifier."""
    base_type = _keyword_detect(text)

    # Memory similarity override
    mem = Memory.latest()
    if mem:
        similar = mem.find_similar(text, top_k=3)
        for entry in similar:
            t = entry.get("metadata", {}).get("type")
            if t in _TaskType.__members__:
                base_type = _TaskType(t)
                break

    # Punctuation heuristic
    if text.strip().endswith("?"):
        base_type = _TaskType.CHAT

    return {"type": base_type.value, "value": text, "confidence": 0.8}

def classify_llm_output(text: str) -> Dict[str, str]:
    """Backwards compatibility wrapper around :func:`classify_intent`."""
    result = classify_intent(text)
    return {"type": result.get("type", "other")}
