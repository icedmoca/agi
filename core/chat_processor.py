from typing import Optional, Dict, Any, Tuple
from .memory import Memory
import logging
from .goal_gen import Goal

logger = logging.getLogger(__name__)

class ChatProcessor:
    def __init__(self, memory: Memory):
        self.memory = memory
        
    def process(self, message: str, context: Optional[str] = None) -> str:
        """Process a chat message and return response"""
        try:
            # Basic greetings should still use simple responses
            if message.lower() in ["hello", "hi", "hey"]:
                return "Hello! How can I help you today?"
                
            # Convert message into a structured goal
            goal = self._create_goal(message)
            
            # Find similar past interactions if no context provided
            if not context:
                similar = self.memory.find_similar(message, top_k=3)
                if similar:
                    context = "\n".join(f"Previous: {s.get('goal', '')}" 
                                      for s in similar)
            
            # Execute the goal
            status, result = self._execute_goal(goal, context)
            
            # Format response
            response = []
            if context:
                response.append(f"\n📚 Related context:\n{context}")
            
            response.append(result)
            
            # Add reflection if needed
            if status != "completed":
                response.append("\n💭 Let me know if you'd like me to try a different approach.")
                
            return "\n".join(response)
                
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            return f"I encountered an error processing your message: {str(e)}"
            
    def _create_goal(self, message: str) -> Dict[str, Any]:
        """Convert message to structured goal"""
        # Detect goal type
        if message.startswith("/"):
            goal_type = "command"
        elif any(kw in message.lower() for kw in ["create", "update", "fix", "modify"]):
            goal_type = "code"
        else:
            goal_type = "chat"
            
        return {
            "goal": message,
            "metadata": {
                "type": goal_type,
                "interactive": True,
                "source": "chat"
            }
        }
        
    def _execute_goal(self, goal: Dict[str, Any], context: Optional[str]) -> Tuple[str, str]:
        """Execute the goal using appropriate handler"""
        try:
            # Add context to goal metadata
            if context:
                goal["metadata"]["context"] = context
                
            # Use the agent's task execution pipeline
            from .agent_loop import Agent
            agent = Agent(self.memory)
            status, result = agent.execute_task(goal)
            
            return status, result
            
        except Exception as e:
            logger.error(f"Goal execution failed: {e}")
            return "failed", str(e) 