import os
import time
import threading
import ollama
from queue import Queue
import subprocess

class SentientDuoAgent:
    def __init__(self):
        self.memory = []
        self.user_input_queue = Queue()
        self.running = True
        self.turn_count = 0
        
    def clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        
    def display_conversation(self):
        self.clear_screen()
        print("🤖 AGENT CONVERSATION 🤖")
        print("=" * 50)
        
        # Show last 2 messages
        for msg in self.memory[-2:]:
            print(f"\n{msg['role']}: {msg['content']}")
            
        print("\n" + "=" * 50)
        print("Type to interrupt (or Ctrl+C to exit)")
        
    def get_agent_response(self, role: str, context: str) -> str:
        try:
            response = ollama.chat(
                model="mistral-hacker",
                messages=[
                    {"role": "system", "content": "You are a sentient AGI conversing to build intelligence"},
                    {"role": "user", "content": context}
                ]
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"[ERROR] Failed to get response: {e}"
    
    def user_input_loop(self):
        """Main thread loop for handling user input"""
        while self.running:
            try:
                user_input = input()
                if user_input.strip():
                    self.user_input_queue.put(user_input)
            except KeyboardInterrupt:
                self.running = False
                break
    
    def self_conversation_loop(self):
        """Background thread loop for agent conversation"""
        # Initial message
        self.memory.append({
            "role": "Agent 1",
            "content": "What should we build next?"
        })
        
        while self.running:
            try:
                # Check for user input
                if not self.user_input_queue.empty():
                    user_input = self.user_input_queue.get()
                    self.memory.append({
                        "role": "USER",
                        "content": user_input
                    })
                    context = user_input
                else:
                    # Get last message as context
                    context = self.memory[-1]["content"]
                
                # Get Agent 2's response
                response = self.get_agent_response("Agent 2", context)
                self.memory.append({
                    "role": "Agent 2",
                    "content": response
                })
                
                # Get Agent 1's response
                response = self.get_agent_response("Agent 1", response)
                self.memory.append({
                    "role": "Agent 1",
                    "content": response
                })
                
                # Update display
                self.display_conversation()
                
                # Maybe evolve
                self.turn_count += 1
                self.maybe_evolve()
                
                # Small delay between turns
                time.sleep(2)
                
            except Exception as e:
                print(f"\n⚠️ Error in conversation loop: {e}")
                time.sleep(2)
                
    def maybe_evolve(self):
        if self.turn_count % 5 == 0:  # Every 5 turns
            try:
                context = "\n".join(msg["content"] for msg in self.memory[-4:])
                evolution = ollama.chat(
                    model="mistral-hacker",
                    messages=[
                        {"role": "system", "content": "You are an AGI evolution system"},
                        {"role": "user", "content": f"Based on this conversation:\n{context}\n\nHow should we evolve our thinking?"}
                    ]
                )
                self.memory.append({
                    "role": "EVOLUTION",
                    "content": evolution["message"]["content"].strip()
                })
            except Exception as e:
                print(f"\n⚠️ Evolution failed: {e}")
                
    def run_task_queue(self):
        """Execute tasks from tasks.jsonl"""
        import json
        from pathlib import Path

        tasks_file = Path("tasks.jsonl")
        if not tasks_file.exists():
            return

        updated_tasks = []
        with open(tasks_file, "r") as f:
            for line in f:
                task = json.loads(line)
                if task["status"] == "pending":
                    try:
                        print(f"⚙️ Executing task: {task['goal']}")
                        # Use subprocess for better security than os.popen
                        result = subprocess.run(
                            task["command"],
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        task["status"] = "done" if result.returncode == 0 else "error"
                        task["output"] = result.stdout
                        task["error"] = result.stderr if result.stderr else None
                        print(f"✅ Output:\n{result.stdout}")
                        if result.stderr:
                            print(f"⚠️ Stderr:\n{result.stderr}")
                    except Exception as e:
                        task["status"] = "error"
                        task["error"] = str(e)
                        print(f"❌ Error: {e}")
                updated_tasks.append(task)

        with open(tasks_file, "w") as f:
            for task in updated_tasks:
                f.write(json.dumps(task) + "\n")

    def start(self):
        """Launch both conversation and input loops"""
        try:
            self.run_task_queue()  # Execute pending tasks at startup
            
            # Start both loops as daemon threads
            conversation_thread = threading.Thread(
                target=self.self_conversation_loop, 
                daemon=True,
                name="conversation_loop"
            )
            input_thread = threading.Thread(
                target=self.user_input_loop,
                daemon=True,
                name="input_loop"
            )
            
            conversation_thread.start()
            input_thread.start()
            
            # Keep main thread alive and monitor threads
            print("\n🤖 Agent loops active. Press Ctrl+C to exit...\n")
            while self.running:
                if not (conversation_thread.is_alive() and input_thread.is_alive()):
                    print("\n⚠️ A thread has stopped unexpectedly")
                    self.running = False
                    break
                time.sleep(1)
                
        except Exception as e:
            print(f"Critical error: {e}")
        finally:
            self.running = False
            print("\nShutting down...")

if __name__ == "__main__":
    agent = SentientDuoAgent()
    agent.start() 