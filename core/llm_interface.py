from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def chat_with_llm(
    user_input: str,  # Renamed from prompt to match expected argument name
    system_prompt: str = "", 
    context: Optional[str] = None,
    memory_context: Optional[List[Dict]] = None
) -> str:
    """Chat with LLM using context-aware prompting
    
    Args:
        user_input: The user's message/goal
        system_prompt: Optional system instructions
        context: Optional string context from memory
        memory_context: Optional structured memory entries
    """
    try:
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Add context if provided
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
            
        # Add memory context if provided
        if memory_context:
            context_str = "\n".join(f"• {m.get('content', '')}" for m in memory_context)
            messages.append({"role": "system", "content": f"Memory:\n{context_str}"})
            
        # Add user input
        messages.append({"role": "user", "content": user_input})
        
        try:
            from ollama import chat
            response = chat(
                model="mistral", 
                messages=messages,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            )
            return response.message.content
            
        except ImportError:
            # Fallback to basic prompt concatenation
            full_prompt = "\n\n".join(m["content"] for m in messages)
            # Call your preferred LLM API here
            return f"LLM would process: {full_prompt}"
            
    except Exception as e:
        logger.error(f"LLM chat failed: {e}")
        return f"I encountered an error processing your request: {str(e)}" 