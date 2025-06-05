import json
import logging
import re
from typing import Dict

from .chat import chat_with_llm

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


def classify_intent(text: str) -> Dict[str, str]:
    """Return a structured intent classification for the given text."""
    text = (text or "").strip()
    if not text:
        return {"type": "other", "value": ""}

    lowered = text.lower()
    for pat in _INTERACTIVE_PATTERNS:
        if re.search(pat, lowered):
            return {"type": "interactive", "value": text, "reason": "Blocks on user input"}

    try:
        response = chat_with_llm(
            user_input=f"Text:\n{text}",
            system_prompt=_CLASSIFY_PROMPT,
            format="json",
            retry_on_invalid=True,
        )
        data = json.loads(response)
        if isinstance(data, dict) and data.get("type"):
            intent_type = str(data.get("type", "other")).lower()
            value = (data.get("value") or text).strip()
            return {"type": intent_type, "value": value}
    except Exception as e:
        logger.error("LLM classification failed: %s", e)

    try:
        from .executor import is_valid_shell_command
        intent_type = "command" if is_valid_shell_command(text) else "other"
    except Exception:
        intent_type = "other"
    return {"type": intent_type, "value": text}


def classify_llm_output(text: str) -> Dict[str, str]:
    """Backwards compatibility wrapper around :func:`classify_intent`."""
    result = classify_intent(text)
    return {"type": result.get("type", "other")}
