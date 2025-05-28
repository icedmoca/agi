python
import time
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
import logging
import sys
import traceback
WATCH_DIR = os.path.abspath(os.getcwd())  # Use absolute path
CHECK_INTERVAL = 3  # seconds
HASH_FILE = "trusted_hashes.txt"
IGNORE_PATTERNS = [
    "venv", "__pycache__", ".git",
    "memory.json", "evolution_log.py",
    "last_error.txt", "hash_sync.log"
]
LOG_FILE = os.path.join(WATCH_DIR, "file_changes.log")
BACKUP_DIR = os.path.join(WATCH_DIR, "backedup")
BACKUP_RETENTION = 5  # Number of backups to keep per file
OUTPUT_DIR = os.path.join(WATCH_DIR, "output")
LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOGGER = logging.getLogger(__name__)
def get_file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        LOGGER.error("Failed to generate file hash:", exc_info=True)
        raise e
def load_hashes():
    hashes = {}
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            for line in f:
                if "::" in line:
                    path, hash = line.strip().split(" :: ")
                    hashes[path] = hash
    return hashes
def save_hashes(hashes):
    os.makedirs("./output", exist_ok=True)
    with open(HASH_FILE, "w") as f:
        for path, h in sorted(hashes.items()):
            if h is not None:
                f.write(f"{path} :: {h}\n")
def should_ignore(path):
    return any(pattern in str(path) for pattern in IGNORE_PATTERNS)
def backup_file(path: Path) -> str:
    """Create timestamped backup of a file"""
    try:
        # Create backup directory
        backup_dir = Path(BACKUP_DIR)
        backup_dir.mkdir(exist_ok=True)
        # Generate backup path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{timestamp}_{path.name}"
        backup_path = backup_dir / backup_name
        # Copy file with metadata
        shutil.copy2(path, backup_path)
        # Cleanup old backups
        cleanup_old_backups(path.name)
        return str(backup_path)
    except Exception as e:
        LOGGER.error("Backup failed for {}: {}".format(path, e))
        raise e
def cleanup_old_backups(filename: str):
    """Keep only recent backups for a file"""
    try:
        backup_dir = Path(BACKUP_DIR)
        backups = list(backup_dir.glob(f"*_{filename}"))
        backups.sort()
        # Remove oldest backups if too many exist
        while len(backups) > BACKUP_RETENTION:
            backups[0].unlink()
            backups.pop(0)
    except Exception as e:
        LOGGER.error("Backup cleanup failed: {}".format(e))
        raise e
def scan_and_detect_change():
    """Scan for file changes with improved error handling"""
    try:
        old_hashes = load_hashes()
        new_hashes = {}
        changes = []
        # Create output directory if needed
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Ensure hash file exists
        if not os.path.exists(HASH_FILE):
            with open(HASH_FILE, "w") as f:
                f.write("# Initial hash file created at {}\n".format(datetime.now().isoformat()))
        for root, _, files in os.walk(WATCH_DIR):
            root_path = Path(root)
            if should_ignore(root_path):
                continue
            for file in files:
                path = root_path / file
                if should_ignore(path):
                    continue
                rel_path = path.relative_to(WATCH_DIR)
                hash = get_file_hash(path)
                if hash:
                    new_hashes[str(rel_path)] = hash
                    if str(rel_path) not in old_hashes:
                        changes.append((str(rel_path), "NEW"))
                    elif old_hashes[str(rel_path)] != hash:
                        changes.append((str(rel_path), "MODIFIED"))
                        # Backup modified files
                        backup_file(path)
        # Check for deletions
        for old_path in old_hashes:
            if old_path not in new_hashes:
                changes.append((old_path, "DELETED"))
        save_hashes(new_hashes)
        if changes:
            # Log changes
            with open(LOG_FILE, "a", encoding='utf-8') as f:
                formatter = logging.Formatter(LOGGING_FORMAT)
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(formatter)
                LOGGER.addHandler(handler)
                LOGGER.info("=== Changes detected at {} ===".format(datetime.now().isoformat()))
                for path, change_type in changes:
                    LOGGER.info("{}: {}".format(change_type, path))
            LOGGER.removeHandler(handler)
        return changes
    except Exception as e:
        LOGGER.error("Scan and detect change failed:", exc_info=True)
        raise e