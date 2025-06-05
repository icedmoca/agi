import platform
import os
import subprocess
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    info = {
        "platform": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    
    # Add uname info if available
    try:
        if hasattr(os, 'uname'):
            info["uname"] = dict(zip(
                ["sysname", "nodename", "release", "version", "machine"],
                os.uname()
            ))
    except Exception as e:
        logger.warning(f"Failed to get uname info: {e}")
        
    # Try to get additional info from shell commands
    try:
        # Get memory info
        if platform.system() != "Windows":
            mem = subprocess.run(["free", "-h"], capture_output=True, text=True)
            if mem.returncode == 0:
                info["memory"] = mem.stdout
                
        # Get disk info
        df = subprocess.run(["df", "-h"], capture_output=True, text=True)
        if df.returncode == 0:
            info["disk"] = df.stdout
            
    except Exception as e:
        logger.warning(f"Failed to get detailed system info: {e}")
        
    return info 