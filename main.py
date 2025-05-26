import time
import random
import ollama
from core.memory import Memory
from core.planner import plan_next_action, analyze_file_change
from core.executor import execute_action
from core.evolver import maybe_evolve
from core.watcher import scan_and_detect_change
from typing import Optional
import os
from core.reward import append_score
import subprocess
from core.vector_memory import VectorMemory
from core.reflection import ReflectionAgent
from datetime import datetime

def analyze_error(error_msg: str) -> str:
    """Analyze error and suggest command fixes"""
    prompt = f"""You are a Linux command debugging expert.
Error: {error_msg}

Suggest ONE specific fix for this error.
- Be precise and technical 
- Focus on command syntax or permissions
- NO general advice or multiple options
- Return ONLY the corrected command
- For file comparison commands, use diff
- Remove backgrounding (&) to prevent race conditions
- Ensure commands complete sequentially"""

    response = ollama.chat(
        model="mistral-hacker", 
        messages=[{"role": "user", "content": prompt}]
    )
    return "timeout 120 find /home -type f -exec sha256sum {} \\; > hashes.txt && diff hashes.txt trusted_hashes.txt > results.txt"

def analyze_memory_for_errors(memory: Memory, limit: int = 5) -> Optional[str]:
    """Check recent memory entries for errors that need attention"""
    recent = memory.get_recent(limit)
    
    for entry in recent:
        result = entry.get('result', '').lower()
        if any(x in result for x in ['error', 'failed', '[stderr]', 'exception']):
            return f"Fix error: {entry['goal']}"
    
    return None

def run_tests():
    """Run unit tests with proper module paths"""
    try:
        # Run tests using module notation
        result = subprocess.run(
            ["python3", "-m", "unittest", "discover", "-s", "tests"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Tests passed")
            return True
        else:
            print(f"❌ Tests failed:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    print("\n🚀 AGI Main Loop Online – Monitoring & Adapting...")
    print("📁 Working directory:", "/home/astro/agi")
    print("⏳ Starting main loop...\n")
    
    # Initialize memories
    memory = Memory("memory.json")
    vector_memory = VectorMemory()
    reflector = ReflectionAgent(vector_memory)
    
    # Load vector memory if exists
    try:
        vector_memory.load()
        print("✅ Loaded vector memory")
    except Exception as e:
        print(f"⚠️ Creating new vector memory: {e}")
    
    # Generate initial hash file if needed
    if not os.path.exists("trusted_hashes.txt"):
        print("⚙️ Generating initial file hashes...")
        subprocess.run(
            ["find", ".", "-type", "f", "!", "-name", "*.md5",
             "-exec", "md5sum", "{}", ";"],
            stdout=open("trusted_hashes.txt", "w")
        )
    
    # Initialize git if needed
    if not os.path.exists(".git"):
        print("📦 Initializing git repository...")
        os.system("git init")
        os.system("git add .")
        os.system('git commit -m "Initial commit"')
    
    while True:
        try:
            # 1. Check for file changes
            changes = scan_and_detect_change()
            if changes:
                for path, change_type in changes:
                    print(f"⚠️ Detected {change_type} file: {path}")
                    
                    # Store in vector memory
                    vector_memory.add(f"File {change_type}: {path}")
                    
                    # Get intelligent analysis
                    decision = analyze_file_change(path, change_type)
                    print(f"🧠 Analysis: {decision}")
                    
                    # Search for related memories
                    related = vector_memory.search(f"{path} {change_type}")
                    if related:
                        print("🔍 Related changes:", related[:2])
                    
                    # Log the analysis
                    memory.append(
                        goal=f"Analyze file change",
                        result=f"{decision}"
                    )
                    
                    # Update scores
                    append_score()
                    
                    # Handle sensitive files specially
                    if "SENSITIVE:" in decision:
                        print(f"\n🎯 Executing goal: Handle sensitive file change")
                        result = execute_action(f"Secure and audit {path}")
                        print(f"✅ Result: {result}")
                        memory.append(
                            goal="Handle sensitive file change",
                            result=result
                        )
                    # Handle code changes
                    elif "CODE CHANGE:" in decision:
                        print(f"\n🎯 Executing goal: Validate code change")
                        result = execute_action(f"Run tests for {path}")
                        print(f"✅ Result: {result}")
                        memory.append(
                            goal="Validate code change",
                            result=result
                        )
                    # Handle config changes
                    elif "CONFIG CHANGE:" in decision:
                        print(f"\n🎯 Executing goal: Verify config")
                        result = execute_action(f"Validate config syntax in {path}")
                        print(f"✅ Result: {result}")
                        memory.append(
                            goal="Verify config",
                            result=result
                        )
            
            # 2. Get recent memory summary and check for errors
            context = memory.get_recent_summary() if hasattr(memory, "get_recent_summary") else ""
            error_goal = analyze_memory_for_errors(memory)
            
            # 3. Plan next action - prioritize error handling
            if error_goal:
                print(f"\n⚠️ Prioritizing error: {error_goal}")
                goal = error_goal
            else:
                goal = plan_next_action(context)
            
            print(f"\n🎯 Executing goal: {goal}")
            result = execute_action(goal)
            print(f"✅ Result: {result}")
            
            # 4. Record the action in memory
            memory.append(goal=goal, result=result)
            
            # Save vector memory periodically
            vector_memory.save()
            
            # After handling changes, reflect on recent activity
            reflection = reflector.reflect(window=30)
            print(f"\n{reflection}\n")
            
            # Periodic reflection even without changes
            if time.time() % 300 < 1:  # Every 5 minutes
                reflection = reflector.reflect(window=50)
                print(f"\n{reflection}\n")
                
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n👋 Shutting down gracefully...")
            vector_memory.save()  # Save before exit
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    print("\n🔄 Starting AGI autonomous loop")
    print("📝 Logging to: loop.log")
    print("⌨️  Press Ctrl+C to stop\n")
    
    while True:
        try:
            main()
            time.sleep(15)  # Polling interval
            
            # Run reflection periodically
            if hasattr(reflector, 'reflect'):
                reflection = reflector.reflect(window=30)
                print(f"\n{reflection}\n")
                
        except KeyboardInterrupt:
            print("\n🛑 Loop terminated manually")
            break
        except Exception as e:
            print(f"\n❌ Loop error: {e}")
            # Log the error
            with open("output/loop.log", "a") as f:
                f.write(f"[{datetime.now()}] Error: {str(e)}\n")
            time.sleep(5)  # Brief delay before retry 
