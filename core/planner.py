# Importing essential libraries for this project
import os
import pathlib
import logging
from pathlib import Path
import ollama
# Initializing logger for debugging and tracking purposes
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')
# Defining project constants and variables
PROJECT_DIR = os.path.abspath(os.getcwd())  # Current working directory (project path)
OUTPUT_DIRECTORY = os.path.join(PROJECT_DIR, "output")  # Output directory for results
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
SENSITIVE_PATTERNS = ["api_key", "secret", "password", "credential", ".env"]  # Sensitive patterns to watch out for
VALID_CODE_EXTENSIONS = ['.py', '.js', '.cpp', '.h']  # Valid code file extensions
VALID_CONFIG_EXTENSIONS = [".yml", ".json", ".toml", ".ini"]  # Valid configuration file extensions
LOG_FILE_EXTENSION = ".log"  # Extension for log files
def analyze_file_change(file_path: str, change_type: str) -> str:
    """Analyze changes in a file and determine appropriate response"""
    analyzed_file_path = Path(file_path).resolve().as_posix()  # Replace "project" with the project directory path
    if not Path(analyzed_file_path).is_file():
        logging.error(f"Error: File {analyzed_file_path} does not exist.")
        return f'File "{analyzed_file_path}" not found.'
    sensitive_found = any(pattern in analyzed_file_path.lower() for pattern in SENSITIVE_PATTERNS)
    if sensitive_found:
        logging.warning(f"Sensitive pattern '{analyzed_file_path}' was {change_type}. Requires secure handling and audit logging.")
        return f'File "{analyzed_file_path}" contains sensitive patterns. Secure handling and audit logging required.' + '\nAudit log created.'
    file_extension = Path(analyzed_file_path).suffix.lower()
    if file_extension in VALID_CODE_EXTENSIONS:
        logging.info(f"Code change detected in '{analyzed_file_path}'. Run tests and validate syntax.")
        return f'Code change detected in "{analyzed_file_path}". Run tests and validate syntax.' + '\nRun tests and validate syntax.'
    elif file_extension in VALID_CONFIG_EXTENSIONS:
        logging.info(f"Configuration change detected in '{analyzed_file_path}'. Verify syntax and permissions.")
        return f'Configuration change detected in "{analyzed_file_path}". Verify syntax and permissions.' + '\nVerify syntax and permissions.'
    elif file_extension == LOG_FILE_EXTENSION:
        logging.info(f"Log update detected in '{analyzed_file_path}'. Check for anomalies.")
        return f'Log update detected in "{analyzed_file_path}". Check for anomalies.' + '\nCheck for anomalies.'
    else:
        logging.warning(f"Generic change detected in '{analyzed_file_path}'. Standard backup and monitoring.")
        return f'Generic change detected in "{analyzed_file_path}". Standard backup and monitoring.' + '\nStandard backup and monitoring.'
def plan_next_action(context: str) -> str:
    """Plan the next action based on memory context using AI"""
    # Logging for debugging and traceability
    logging.info("Next action planning context: {}".format(context))
    prompt = f"""\
You are an autonomous AI with root access to a Linux system.
Here is what you recently did:
{context}
Now decide: what is a practical, meaningful action you can take next?
 DO NOT be funny.
 DO NOT use sarcasm.
 DO NOT return junk goals like reading /dev/urandom.
 DO return a specific, safe, useful system task (e.g., find files, list packages, scan directories, summarize logs).
 DO NOT fix previous failed commands unless explicitly asked.
 NEVER re-run broken goals automatically.
Return only a single sentence goal with no jokes, no quotes, no formatting.
"""
    response = ollama.chat(
        model="mistral-hacker",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()