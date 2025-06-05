from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path
import shutil
import os
from core import planner, evolver, executor
from core.memory import Memory

WATCH_DIR = os.path.abspath(os.getcwd())
BACKUP_DIR = os.path.join(WATCH_DIR, "backedup")
OUTPUT_DIR = os.path.join(WATCH_DIR, "output")
THROTTLE_SECONDS = 5
MAX_MEMORY_ENTRIES = 100

IGNORE_PATTERNS = [
    "venv", "__pycache__", ".git",
    "memory.json", "evolution_log.py",
    "last_error.txt", "hash_sync.log",
    "*.pyc", "*.pyo", "*.pyd", "*.so",
    "/backedup/",  # Prevent recursive backups
    "/output/"     # Don't watch output directory
]

class IntelligentChangeHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.last_events = {}
        self.memory = Memory("memory.json")
        self.setup_directories()
        
    def setup_directories(self):
        """Ensure all required directories and files exist"""
        # Create directories
        for dir in [BACKUP_DIR, OUTPUT_DIR]:
            os.makedirs(dir, exist_ok=True)
            
        # Create baseline files if needed
        baseline_files = {
            "trusted_hashes.txt": "# Initial trusted hashes\n",
            os.path.join(OUTPUT_DIR, "hash_check_errors.log"): "",
            os.path.join(OUTPUT_DIR, "file_changes.log"): ""
        }
        
        for filepath, initial_content in baseline_files.items():
            if not os.path.exists(filepath):
                with open(filepath, "w") as f:
                    f.write(f"# Created at {datetime.now().isoformat()}\n{initial_content}")
            
    def should_ignore(self, path: str) -> bool:
        """Enhanced ignore check"""
        path_str = str(path)
        return any(pattern in path_str for pattern in IGNORE_PATTERNS) or \
               path_str.startswith(('./backedup/', './output/'))
               
    def is_throttled(self, path: str) -> bool:
        """Check if file was recently processed"""
        now = time.time()
        if path in self.last_events:
            if now - self.last_events[path] < THROTTLE_SECONDS:
                return True
        self.last_events[path] = now
        return False
        
    def validate_execution(self, path: str, output: str, error: str = None) -> dict:
        """Validate execution results and prepare memory entry"""
        timestamp = datetime.now().isoformat()
        result = "success" if not error else "error"
        
        # Write to appropriate log file
        log_file = "last_success.txt" if result == "success" else "last_error.txt"
        with open(os.path.join(OUTPUT_DIR, log_file), "w") as f:
            f.write(f"=== {timestamp} ===\n")
            f.write(f"File: {path}\n")
            f.write(f"Output:\n{output}\n")
            if error:
                f.write(f"Error:\n{error}\n")
                
        # Get analysis from planner
        analysis = planner.analyze_file_change(path, "EXECUTED")
        
        return {
            "timestamp": timestamp,
            "filename": path,
            "change_type": "EXECUTED",
            "result": result,
            "summary": analysis
        }

    def handle_code_execution(self, path: str):
        """Execute and validate code changes"""
        try:
            # Execute file and capture output
            result = executor.run_and_capture(["python", path])
            
            # Validate results
            memory_entry = self.validate_execution(
                path=path,
                output=result.stdout,
                error=result.stderr if result.stderr else None
            )
            
            # Update memory
            self.memory.append(
                goal=f"Execute and validate {path}",
                result=json.dumps(memory_entry)
            )
            
            # If error occurred, trigger evolution
            if memory_entry["result"] == "error":
                print(f"⚠️ Execution failed, attempting evolution")
                try:
                    evolver.evolve_file(path)
                    print(f"🧬 Evolution triggered for {path}")
                except Exception as e:
                    print(f"❌ Evolution failed: {e}")
                    
        except Exception as e:
            print(f"❌ Execution failed: {e}")
            
    def analyze_code_change(self, path: str, event_type: str):
        """Analyze code changes and potentially trigger evolution"""
        try:
            # Get intelligent analysis
            decision = planner.analyze_file_change(path, event_type)
            print(f"🧠 ANALYZED: {path}")
            
            # Check if change warrants evolution
            if "evolution" in decision.lower() or "improve" in decision.lower():
                try:
                    evolver.evolve_file(path)
                    print(f"🧬 EVOLVED: {path}")
                except Exception as e:
                    print(f"⚠️ Evolution failed for {path}: {e}")
                    
            return decision
        except Exception as e:
            print(f"❌ Analysis failed for {path}: {e}")
            return None
            
    def handle_change(self, event_type: str, path: str):
        """Enhanced change handler with validation and memory"""
        if self.should_ignore(path) or self.is_throttled(path):
            return
            
        try:
            path_obj = Path(path)
            
            # Handle main.py specially
            if path_obj.name == "main.py":
                print(f"🔄 Executing main.py")
                self.handle_code_execution(path)
                
            # Handle core/ and agents/ files
            elif any(part in str(path_obj) for part in ['/core/', '/agents/']):
                decision = self.analyze_code_change(path, event_type)
                if decision:
                    # Check if change relates to current goal
                    current_goal = self.get_current_goal()
                    if current_goal and any(word in decision.lower() for word in current_goal.lower().split()):
                        print(f"🎯 Change relates to current goal")
                        self.handle_code_execution(path)
                    
                    self.log_event(event_type, path, decision)
                    
            # Backup if appropriate
            if event_type in ["MODIFIED", "CREATED"] and os.path.exists(path):
                backup_path = self.backup_file(path)
                if backup_path:
                    print(f"📦 BACKED UP: {path}")
                    
        except Exception as e:
            print(f"❌ Error handling {path}: {e}")
            self.log_event(event_type, path, f"ERROR: {str(e)}")
            
    def backup_file(self, path: str) -> str:
        """Create timestamped backup of file"""
        try:
            if "/backedup/" in str(path) or "/output/" in str(path):
                return None
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{timestamp}_{os.path.basename(path)}"
            backup_path = os.path.join(BACKUP_DIR, backup_name)
            shutil.copy2(path, backup_path)
            return backup_path
        except Exception as e:
            print(f"⚠️ Backup failed for {path}: {e}")
            return None
            
    def log_event(self, event_type: str, path: str, details: str = None):
        """Log file system events with details"""
        try:
            with open(os.path.join(OUTPUT_DIR, "realtime_changes.log"), "a") as f:
                timestamp = datetime.now().isoformat()
                f.write(f"\n=== {event_type} at {timestamp} ===\n")
                f.write(f"Path: {path}\n")
                if details:
                    f.write(f"Details: {details}\n")
        except Exception as e:
            print(f"⚠️ Logging failed: {e}")
            
    def get_current_goal(self) -> str:
        """Get current goal from memory"""
        try:
            with open("memory.json", "r") as f:
                data = json.load(f)
                return data.get("current_goal", "")
        except:
            return ""
            
    def on_modified(self, event):
        if not event.is_directory:
            self.handle_change("MODIFIED", event.src_path)
            
    def on_created(self, event):
        if not event.is_directory:
            self.handle_change("CREATED", event.src_path)
            
    def on_deleted(self, event):
        if not event.is_directory:
            self.handle_change("DELETED", event.src_path)

def start_watcher():
    """Start the intelligent file system watcher"""
    observer = Observer()
    handler = IntelligentChangeHandler()
    observer.schedule(handler, WATCH_DIR, recursive=True)
    observer.start()
    
    print("\n🤖 Intelligent File Watcher Active")
    print(f"📁 Monitoring: {os.path.abspath(WATCH_DIR)}")
    print(f"💾 Backups: {os.path.abspath(BACKUP_DIR)}")
    print(f"📝 Logs: {os.path.abspath(OUTPUT_DIR)}")
    print("⏳ Throttle: 5 seconds per file")
    print("\nPress Ctrl+C to stop...\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping watcher...")
        observer.stop()
    observer.join()
    print("✅ Watcher stopped cleanly")

if __name__ == "__main__":
    start_watcher() 