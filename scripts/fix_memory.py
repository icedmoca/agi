#!/usr/bin/env python3
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_memory_file(filename: str = "memory.jsonl") -> None:
    """Add metadata field to memory entries that are missing it"""
    memory_path = Path(filename)
    if not memory_path.exists():
        logger.warning(f"Memory file {filename} not found")
        return
        
    # Read all entries
    entries = []
    fixed_count = 0
    
    logger.info(f"Reading {filename}...")
    with memory_path.open() as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line)
                    if "metadata" not in entry:
                        entry["metadata"] = {}
                        fixed_count += 1
                    entries.append(entry)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON on line {line_num}")
                    continue
                    
    # Write back all entries
    logger.info(f"Writing {len(entries)} entries back to {filename}...")
    with memory_path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    logger.info(f"Fixed {fixed_count} entries missing metadata")

if __name__ == "__main__":
    fix_memory_file() 