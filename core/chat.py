import ollama
import logging
from typing import Optional, Dict, Any, List, Callable, Union, Iterator
from pathlib import Path
import json
from datetime import datetime
import re
from functools import partial
import yaml
from html.parser import HTMLParser
import html

logger = logging.getLogger(__name__)
current_system_prompt = "You are a helpful AI assistant."

def get_system_prompt() -> str:
    return current_system_prompt

def set_system_prompt(prompt: str) -> None:
    global current_system_prompt
    current_system_prompt = prompt


class ChatConfig:
    """Configuration for chat functionality"""
    DEFAULT_MODEL = "mistral"
    DEFAULT_TEMPERATURE = 0.7
    TIMEOUT = 60
    MAX_RETRIES = 3
    
    # Model-specific settings
    MODEL_CONFIGS = {
        "mistral": {"temperature": 0.7},
        "codellama": {"temperature": 0.8},
        "llama2": {"temperature": 0.7}
    }

def clean_plain(response: str) -> str:
    """Clean plain text responses"""
    # Remove markdown and links
    response = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', response)
    response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
    
    # Remove search boilerplate
    response = re.sub(r"Search results for.*$", "", response, flags=re.MULTILINE | re.DOTALL)
    response = re.sub(r"Here are some relevant results:.*$", "", response, flags=re.MULTILINE | re.DOTALL)
    response = re.sub(r"Based on the search results,?\s*", "", response)
    
    # Normalize whitespace
    return ' '.join(response.split())

def clean_json(response: str) -> str:
    """Extract and validate JSON from response"""
    try:
        # More robust JSON detection
        json_match = re.search(r'```json\s*([\s\S]+?)\s*```|({[\s\S]+}|\[[\s\S]+\])', response)
        if not json_match:
            raise ValueError("No JSON pattern found")
            
        json_str = json_match.group(1) or json_match.group(2)
        
        # Try to parse and re-format
        parsed = json.loads(json_str)
        return json.dumps(parsed, indent=2)
        
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON, attempting repair: {e}")
        try:
            # Basic JSON repair attempts
            fixed = re.sub(r'(?<!\\)"(\w+)"(?=\s*:)', r'"\1"', json_str)  # Fix unquoted keys
            fixed = re.sub(r',\s*([\]}])', r'\1', fixed)  # Remove trailing commas
            parsed = json.loads(fixed)
            logger.info("JSON repair successful")
            return json.dumps(parsed, indent=2)
        except Exception as e:
            logger.error(f"JSON repair failed: {e}")
            return "{}"  # Return empty JSON as fallback
            
    except Exception as e:
        logger.error(f"JSON extraction failed: {e}")
        return "{}"

def clean_code(response: str) -> str:
    """Clean and format code blocks"""
    # Remove code fences
    code = re.sub(r'```\w*\n?|\n?```', '', response)
    
    # Split into lines and process
    lines = code.splitlines()
    cleaned_lines = []
    
    # Track minimum indent
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    # Normalize indentation and remove empty lines
    for line in lines:
        if line.strip():
            # Remove common indentation prefix
            if min_indent < float('inf'):
                line = line[min_indent:]
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def score_llm_output(response: str, format: str = "plain") -> float:
    """Score LLM output quality based on format-specific heuristics"""
    score = 0.5  # Base score
    
    # Length penalties
    if len(response.strip()) < 10:
        score -= 0.3
    
    # Negative patterns
    if any(p in response.lower() for p in [
        "search results", 
        "i couldn't find",
        "i don't know",
        "i am unable to"
    ]):
        score -= 0.2
        
    # Format-specific scoring
    if format == "json":
        try:
            json.loads(response)
            score += 0.3
        except:
            score -= 0.3
    elif format == "yaml":
        try:
            yaml.safe_load(response)
            score += 0.3
        except:
            score -= 0.3
    elif format == "code":
        if re.search(r'^\s*(?:def|class|import|from)\s', response, re.M):
            score += 0.2
            
    return max(0.0, min(1.0, score))

def clean_markdown(response: str) -> str:
    """Clean markdown while preserving headers and emphasis"""
    # Keep headers and emphasis, remove code blocks
    response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
    response = re.sub(r'`.*?`', '', response)
    
    # Preserve headers and emphasis
    response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Bold
    response = re.sub(r'\*(.*?)\*', r'\1', response)      # Italic
    
    return response.strip()

def clean_html(response: str) -> str:
    """Clean and sanitize HTML"""
    # Remove script tags and their contents
    response = re.sub(r'<script.*?>.*?</script>', '', response, flags=re.DOTALL)
    
    # Basic HTML sanitization
    response = html.escape(response)
    
    # Parse and clean
    parser = HTMLParser()
    parser.feed(response)
    return parser.get_data()

def clean_yaml(response: str) -> str:
    """Extract and validate YAML"""
    try:
        # Remove markdown wrapper if present
        yaml_match = re.search(r'```ya?ml\s*([\s\S]+?)\s*```|([\s\S]+)', response)
        if not yaml_match:
            raise ValueError("No YAML found")
            
        yaml_str = yaml_match.group(1) or yaml_match.group(2)
        
        # Validate by parsing
        parsed = yaml.safe_load(yaml_str)
        return yaml.dump(parsed, default_flow_style=False)
        
    except Exception as e:
        logger.warning(f"YAML parsing failed: {e}")
        return response.strip()

def clean_shell(response: str) -> str:
    """Clean shell commands, keeping only valid syntax"""
    # Remove markdown
    response = re.sub(r'```(?:sh|bash)?\s*([\s\S]+?)\s*```', r'\1', response)
    
    # Keep only valid shell lines
    lines = []
    for line in response.splitlines():
        line = line.strip()
        if line and not line.startswith(('#', '$')):
            lines.append(line)
            
    return '\n'.join(lines)

# Extend format handlers
format_handlers: Dict[str, Callable[[str], str]] = {
    "plain": clean_plain,
    "json": clean_json,
    "code": clean_code,
    "markdown": clean_markdown,
    "html": clean_html,
    "yaml": clean_yaml,
    "shell": clean_shell
}

def sanitize_llm_response(response: str, format: str = "plain") -> str:
    """Sanitize and format LLM responses"""
    if not response:
        return "" if format != "json" else "{}"
        
    # Basic cleanup
    response = response.strip()
    
    # Get appropriate handler
    handler = format_handlers.get(format, clean_plain)
    
    try:
        return handler(response)
    except Exception as e:
        logger.error(f"Response sanitization failed for format {format}: {e}")
        return "" if format != "json" else "{}"

def chat_with_llm(
    user_input: str,
    system_prompt: str | None = None,
    format: str = "plain",
    model: str = "mistral-hacker",
    stream: bool = False,
    retry_on_invalid: bool = True
) -> Union[str, Iterator[str]]:
    """Chat with LLM and format response, with optional streaming"""
    
    format_instructions = {
        "json": "\nRespond with valid JSON only.",
        "code": "\nRespond with clean code only, no markdown.",
        "yaml": "\nRespond with valid YAML only.",
        "html": "\nRespond with valid HTML only.",
        "shell": "\nRespond with valid shell commands only.",
        "markdown": "\nRespond in markdown format.",
        "plain": "\nRespond naturally without markdown or URLs unless specifically requested."
    }
    
    full_prompt = (system_prompt or get_system_prompt()) + format_instructions.get(format, "")
    
    try:
        if stream:
            return _stream_chat(full_prompt, user_input, format, model)
        else:
            return _single_chat(full_prompt, user_input, format, model, retry_on_invalid)
            
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return "" if format != "json" else "{}"

def _stream_chat(
    system_prompt: str,
    user_input: str, 
    format: str,
    model: str
) -> Iterator[str]:
    """Stream chat response chunks"""
    try:
        from ollama import chat
        
        response_chunks = []
        for chunk in chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            stream=True
        ):
            chunk_text = chunk["message"]["content"]
            response_chunks.append(chunk_text)
            yield chunk_text
            
        # Return final sanitized response
        final_response = "".join(response_chunks)
        cleaned = sanitize_llm_response(final_response, format)
        _log_llm_history(system_prompt or get_system_prompt(), user_input, cleaned, model)
        return cleaned
        
    except ImportError:
        yield "[ERROR] Streaming not supported without ollama package"

def _single_chat(
    system_prompt: str,
    user_input: str,
    format: str, 
    model: str,
    retry_on_invalid: bool
) -> str:
    """Single chat interaction with optional retry"""
    from ollama import chat
    
    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    
    result = sanitize_llm_response(response["message"]["content"], format)
    
    # Retry once if format validation failed
    if retry_on_invalid and result in ("", "{}") and format != "plain":
        logger.info(f"Retrying with stricter {format} format instructions")
        stricter_prompt = system_prompt + f"\nYou MUST respond with valid {format} format ONLY."
        
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": stricter_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        result = sanitize_llm_response(response["message"]["content"], format)

    _log_llm_history(system_prompt or get_system_prompt(), user_input, result, model)
    
    return result

def log_chat(
    model: str,
    messages: List[Dict[str, str]], 
    result: str,
    success: bool
) -> None:
    """Log chat interaction to chat_history.jsonl"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "messages": messages,
        "result": result,
        "success": success
    }
    
    log_file = Path("output/chat_history.jsonl")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with log_file.open("a") as f:
        f.write(json.dumps(log_entry) + "\n")

def _log_llm_history(system_prompt: str, user_input: str, output: str, model: str) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "system_prompt": system_prompt,
        "user_input": user_input,
        "output": output,
    }
    log_file = Path("output/llm_history.jsonl")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")
