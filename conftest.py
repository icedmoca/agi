"""
Ensure the project root directory is on sys.path for all tests.

This lets `import core.<module>` succeed even when pytest is launched from an
arbitrary sub-directory.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
