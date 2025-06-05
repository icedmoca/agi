import json
import logging
from typing import Dict

from .chat import chat_with_llm

logger = logging.getLogger(__name__)

_CLASSIFY_PROMPT = (
    "You are an intent classifier. Categorize the given text into one of the"
    " following types: command, explanation, instruction, error, suggestion,"
    " other. Respond ONLY with JSON like {\"type\": \"command\"}."
)


def classify_llm_output(text: str) -> Dict[str, str]:
    """Classify LLM output text into an intent type using the LLM itself."""
    text = (text or "").strip()
    if not text:
        return {"type": "other"}
    try:
        response = chat_with_llm(
            user_input=f"Text:\n{text}",
            system_prompt=_CLASSIFY_PROMPT,
            format="json",
            retry_on_invalid=True,
        )
        data = json.loads(response)
        if isinstance(data, dict) and data.get("type"):
            return {"type": str(data["type"]).lower()}
    except Exception as e:
        logger.error("LLM classification failed: %s", e)
    # Fallback heuristic
    try:
        from .executor import is_valid_shell_command
        return {"type": "command" if is_valid_shell_command(text) else "other"}
    except Exception:
        return {"type": "other"}
