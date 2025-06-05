# NOTE: This is the ONLY main.py. Do not duplicate under core/

import os
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import random
import ollama
import sys
import logging
import atexit
import signal
import warnings
from contextlib import suppress
import traceback
import threading

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from core.executor import execute_action
from core.reflection import ReflectionAgent
from core.vector_memory import VectorMemory
from core.memory import Memory
from core.vector_memory import get_vectorizer
from core.agent_loop import Agent, process_task
from core.audit import AuditLogger
from core.task_sanitizer import TaskSanitizer
from core.goal_gen import Goal, GoalGenerator
from core.tools.fetcher import SystemFetcher
from core.models import Task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = "/home/astro/agi"
LOOP_LOG = f"{PROJECT_ROOT}/output/loop.log"
MEMORY_FILE = f"{PROJECT_ROOT}/memory.json"
PENDING_GOALS = f"{PROJECT_ROOT}/output/pending_goals.json"
CHANGES_LOG = f"{PROJECT_ROOT}/output/file_changes.log"

IMPROVEMENT_TYPES = [
    "improve error handling",
    "optimize performance",
    "enhance input validation",
    "reduce memory usage",
    "increase code clarity",
    "strengthen safety checks",
    "add better logging",
    "improve documentation"
]

print("🧠 AGI Boot Sequence Initiated")

def load_pending_goals() -> List[Dict]:
    """Load pending goals from JSON file"""
    if not os.path.exists(PENDING_GOALS):
        return []
    try:
        with open(PENDING_GOALS, 'r') as f:
            return json.load(f)
    except:
        return []

def save_pending_goals(goals: List[Dict]):
    """Save pending goals to JSON file"""
    os.makedirs(os.path.dirname(PENDING_GOALS), exist_ok=True)
    with open(PENDING_GOALS, 'w') as f:
        json.dump(goals, f, indent=2)

def get_recent_changes() -> List[str]:
    """Get list of recently modified Python files"""
    if not os.path.exists(CHANGES_LOG):
        return []
        
    changes = []
    try:
        with open(CHANGES_LOG, 'r') as f:
            lines = f.readlines()[-50:]  # Last 50 lines
            for line in lines:
                if "MODIFIED:" in line and line.endswith('.py'):
                    file = line.split("MODIFIED:")[-1].strip()
                    if file.startswith(f"{PROJECT_ROOT}/core/"):
                        changes.append(os.path.basename(file))
    except Exception as e:
        log_event(f"Error reading changes log: {e}", "error")
    return changes

def generate_evolution_goals(memory: VectorMemory) -> List[Dict]:
    """Generate new evolution goals based on system state"""
    goals = []
    
    # Check recently modified files
    changed_files = get_recent_changes()
    
    # Search memory for problematic files
    problem_files = set()
    for event in memory.search("error failed problem issue bug", top_k=10):
        for word in event.split():
            if word.endswith('.py'):
                problem_files.add(word)
                
    # Combine both sources
    target_files = set(changed_files) | problem_files
    
    # Generate goals for each file
    for file in target_files:
        if not file.endswith('.py') or not os.path.exists(f"{PROJECT_ROOT}/core/{file}"):
            continue
            
        # Pick random improvement type
        improvement = random.choice(IMPROVEMENT_TYPES)
        
        goals.append({
            "id": f"evolve_{int(time.time())}",
            "type": "evolution",
            "goal": f"evolve {file} to {improvement}",
            "created": datetime.now().isoformat(),
            "status": "pending"
        })
        
    return goals

def log_event(message: str, goal_type: Optional[str] = None):
    """Log events with timestamps and type"""
    os.makedirs(os.path.dirname(LOOP_LOG), exist_ok=True)
    timestamp = datetime.now().isoformat()
    type_tag = f"[{goal_type}]" if goal_type else ""
    
    with open(LOOP_LOG, "a") as f:
        f.write(f"{timestamp} {type_tag} {message}\n")

def handle_goal(goal: Dict) -> str:
    """Process a goal and return the result"""
    try:
        # Log goal start
        log_event(f"Starting: {goal['goal']}", goal['type'])
        
        # Execute the goal
        result = execute_action(goal['goal'])
        
        # Log outcome
        success = "[SUCCESS]" in result or "✅" in result
        status = "succeeded" if success else "failed"
        log_event(f"Goal {status}: {result}", goal['type'])
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing goal: {str(e)}"
        log_event(error_msg, "error")
        return f"[ERROR] {error_msg}"

def research_topic(topic: str, memory: VectorMemory) -> str:
    """Use Ollama to research a topic and store insights"""
    prompt = f"""Research the following topic and provide key insights:
{topic}

Format your response as:
SUMMARY: Brief overview
INSIGHTS:
- Key point 1
- Key point 2
APPLICATIONS:
- Potential use 1
- Potential use 2
"""
    
    try:
        response = ollama.chat(
            model="mistral-hacker",
            messages=[{"role": "user", "content": prompt}],
            options={"timeout": 60}
        )
        
        insights = response["message"]["content"]
        
        # Store in memory
        memory.add(f"Research on {topic}:\n{insights}")
        
        return f"✅ Research completed and stored:\n{insights}"
        
    except Exception as e:
        return f"[ERROR] Research failed: {str(e)}"

def update_stub_history(filename: str, goal: str, status: str, error: Optional[str] = None):
    """Update stub evolution history"""
    history_file = f"{PROJECT_ROOT}/output/stub_history.jsonl"
    
    # Read existing history
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = [json.loads(line) for line in f]
    
    # Find and update entry
    for entry in history:
        if entry["filename"] == filename:
            evolution_attempt = {
                "timestamp": datetime.now().isoformat(),
                "goal": goal,
                "status": status
            }
            if error:
                evolution_attempt["error"] = error
            
            entry["evolution_attempts"].append(evolution_attempt)
            entry["status"] = status
            break
    
    # Write updated history
    with open(history_file, 'w') as f:
        for entry in history:
            f.write(json.dumps(entry) + "\n")
            
    # Mark as implemented in memory if successful
    if status == "evolved":
        memory = VectorMemory()
        memory.add(f"Implemented stub: {filename}")

def check_stub_priority(filename: str) -> bool:
    """Check if stub should be evolved based on history"""
    history_file = f"{PROJECT_ROOT}/output/stub_history.jsonl"
    if not os.path.exists(history_file):
        return True
        
    with open(history_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry["filename"] == filename:
                # Count recent failures
                recent_failures = 0
                cooldown_time = datetime.now() - timedelta(hours=24)
                
                for attempt in entry.get("evolution_attempts", []):
                    attempt_time = datetime.fromisoformat(attempt["timestamp"])
                    if attempt["status"] == "failed" and attempt_time > cooldown_time:
                        recent_failures += 1
                
                # Skip if too many recent failures
                if recent_failures >= 3:
                    print(f"⚠️ Skipping {filename} due to {recent_failures} recent failed evolutions")
                    return False
                    
    return True

def check_and_evolve_stubs(memory: VectorMemory) -> Optional[Goal]:
    """Search for stubs and create evolution goals"""
    # Search memory for stub creations and implementations
    stub_events = memory.search("Created stub:", top_k=10)
    
    # Debug what memory returns
    print("\n🔍 Checking stubs:")
    print("DEBUG: Implemented stub events =", memory.search("Implemented stub:", top_k=50))
    
    # Safer implementation status checking
    implemented = set()
    for event in memory.search("Implemented stub:", top_k=50):
        if "Implemented stub:" in event:
            parts = event.split("Implemented stub:")
            if len(parts) > 1:
                implemented.add(parts[1].strip())
    
    log_event(f"Found {len(implemented)} implemented stubs", "debug")
    
    for event in stub_events:
        # Extract script path and goal
        if "Created stub:" in event and "for goal:" in event:
            parts = event.split("for goal:")
            script_info = parts[0].replace("Created stub:", "").strip()
            filename = os.path.basename(script_info)
            
            # Skip if already implemented or low priority
            if filename in implemented or not check_stub_priority(filename):
                log_event(f"Skipping {filename} - implemented={filename in implemented}", "debug")
                continue
                
            original_goal = parts[1].strip()
            
            # Check if file still exists and is a stub
            if os.path.exists(script_info):
                with open(script_info, 'r') as f:
                    content = f.read()
                    if "TODO: Implement" in content and "[STUB]" in content:
                        log_event(f"Found evolvable stub: {filename}", "debug")
                        # Create evolution goal
                        return Goal(
                            type="evolution",
                            description=f"evolve {filename} to implement {original_goal}",
                            parent_id=None,
                            source="stub"
                        )
    return None

def process_goal(goal: Goal, memory: VectorMemory) -> str:
    """Process a goal and its subgoals recursively"""
    try:
        print(f"\n🔍 Processing goal:")
        print(f"Type: {goal.type}")
        print(f"Description: {goal.description}")
        print(f"Initial status: {goal.status}")
        
        # Handle research goals
        if goal.type == "research":
            result = research_topic(goal.description, memory)
            print(f"DEBUG: Research result = {result}")
            goal.result = result
            goal.status = "completed" if "✅" in result else "failed"
            print(f"DEBUG: Goal after research: status={goal.status}, result={goal.result}")
            return result
            
        # Handle evolution goals
        if goal.type == "evolution":
            # Extract filename if this is a stub evolution
            if "evolve " in goal.description and " to implement " in goal.description:
                parts = goal.description.split(" to implement ")
                filename = parts[0].replace("evolve ", "").strip()
                implementation_goal = parts[1].strip()
                
                print(f"DEBUG: Evolving stub {filename} with goal: {implementation_goal}")
                result = execute_action(goal.description)
                print(f"DEBUG: Evolution result = {result}")
                
                # Update stub history based on result
                status = "evolved" if "[SUCCESS]" in result else "failed"
                error = result if "[ERROR]" in result else None
                print(f"DEBUG: Updating stub history: status={status}, error={error}")
                update_stub_history(filename, implementation_goal, status, error)
                
                goal.result = result
                goal.status = "completed" if "[SUCCESS]" in result else "failed"
                print(f"DEBUG: Goal after evolution: status={goal.status}, result={goal.result}")
                return result
            
            # Generate potential subgoals for complex evolutions
            if not goal.subgoals and "complex" in goal.description.lower():
                goal.subgoals = [
                    Goal("research", f"Research best practices for {goal.description}"),
                    Goal("evolution", f"evolve {goal.description} phase 1: basic improvements"),
                    Goal("evolution", f"evolve {goal.description} phase 2: advanced features")
                ]
            
        # Process subgoals first
        for subgoal in goal.subgoals:
            if subgoal.status == "pending":
                subgoal_result = process_goal(subgoal, memory)
                log_event(f"Subgoal {subgoal.id}: {subgoal_result}", subgoal.type)
                
                # Stop if subgoal failed
                if "[ERROR]" in subgoal_result:
                    goal.status = "failed"
                    goal.result = f"Failed due to subgoal: {subgoal_result}"
                    return goal.result
        
        # Execute main goal
        result = execute_action(goal.description)
        goal.result = result
        goal.status = "completed" if "[SUCCESS]" in result or "✅" in result else "failed"
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing goal: {str(e)}"
        goal.status = "failed"
        goal.result = error_msg
        return f"[ERROR] {error_msg}"

def save_goals(goals: List[Goal]):
    """Save goals to JSON file"""
    os.makedirs(os.path.dirname(PENDING_GOALS), exist_ok=True)
    with open(PENDING_GOALS, 'w') as f:
        json.dump([g.to_dict() for g in goals], f, indent=2)

def load_goals() -> List[Goal]:
    """Load goals from JSON file"""
    if not os.path.exists(PENDING_GOALS):
        return []
    try:
        with open(PENDING_GOALS, 'r') as f:
            data = json.load(f)
            return [Goal.from_dict(g) for g in data]
    except:
        return []

def setup_agent() -> Agent:
    """Initialize agent with proper memory and vectorizer"""
    try:
        # Don't pass vectorizer directly to Memory
        memory = Memory(filename=MEMORY_FILE, use_vectorizer=True)
        logger.info("Memory system initialized")
    except Exception as e:
        logger.error(f"Memory initialization failed: {e}")
        memory = Memory(filename=MEMORY_FILE, use_vectorizer=False)
        
    return Agent(memory=memory)

def interactive_mode(agent: Agent):
    """Run interactive REPL mode"""
    logger.info("🤖 Interactive mode active")
    print("\n🤖 AGI Interactive Shell")
    print("Type 'exit' or press Ctrl+C to quit")
    print("Type 'help' for available commands\n")
    
    while True:
        try:
            print("🤖 >>> ", end="", flush=True)
            user_input = input().strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  help     - Show this help message")
                print("  status   - Show agent status")
                print("  clear    - Clear the screen")
                print("  exit     - Exit the shell\n")
                continue
                
            # Instantiate a real Task object
            task = Task(
                id=f"task_{int(time.time())}",
                type="interactive",
                goal=user_input,
                priority=0,
                metadata={"type": "chat", "interactive": True},
            )

            # Dispatch and receive the populated Task back
            task = agent.process_task(task)
            print(f"\n{task.result}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"\n❌ Error: {str(e)}\n")

def cleanup():
    """Cleanup resources before exit"""
    logger.info("Cleaning up resources...")
    # Add any cleanup logic here

def handle_interrupt(signum, frame):
    """Handle interrupt signal gracefully"""
    cleanup()
    sys.exit(0)

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Shutting down agent...")
    sys.exit(0)

def background_loop(agent: Agent):
    """Run continuous background monitoring"""
    while True:
        try:
            agent.run_background_tasks()
        except Exception as e:
            logger.error(f"Background loop error: {e}")
            if "permission denied" in str(e).lower():
                agent.is_safe_mode = True
        time.sleep(60)  # Base interval

def setup_logging():
    """Setup logging with proper handlers and cleanup"""
    # Create handlers with proper encoding
    file_handler = logging.FileHandler('app.log', encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatters and add it to handlers
    log_format = '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger and remove existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    
    # Add new handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return file_handler, console_handler

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup logging with proper cleanup
    file_handler, console_handler = setup_logging()
    
    try:
        # Initialize components
        memory = Memory()
        agent = Agent(memory)
        
        logger.info("Starting agent monitoring system...")
        
        # Start background monitoring thread
        monitor_thread = threading.Thread(
            target=background_loop,
            args=(agent,),
            daemon=True
        )
        monitor_thread.start()
        logger.info("Background monitoring started")
        
        # Run in interactive mode if requested
        if args.interactive:
            while True:
                goal = input("🤖  Goal > ").strip()
                if not goal:
                    continue
                if goal.lower() in {"exit", "quit"}:
                    break

                task = Task(id=str(time.time_ns()), type="chat", goal=goal)
                updated = process_task(task)
                print(updated.result)
            return
            
        # Main monitoring loop
        last_task_check = datetime.now()
        task_interval = timedelta(minutes=15)
        
        while True:
            # Process scheduled tasks
            now = datetime.now()
            if now - last_task_check > task_interval:
                try:
                    agent.process_pending_tasks()
                    last_task_check = now
                except Exception as e:
                    logger.error(f"Task processing error: {e}")
            
            # Adaptive sleep based on system state
            sleep_time = 60  # Default 1-minute pulse
            if agent.is_safe_mode:
                sleep_time = 30  # More frequent checks in safe mode
            elif any(a.get('severity') == 'critical' 
                    for a in agent.recent_alerts.values()):
                sleep_time = 15  # More frequent during issues
                
            time.sleep(sleep_time)
            
    except Exception as e:
        logger.error(f"Main loop error: {e}")
        raise
    finally:
        # Cleanup logging handlers
        file_handler.close()
        console_handler.close()
        logger.info("Agent shutting down...")

if __name__ == "__main__":
    main() 