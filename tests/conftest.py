"""Configure test environment: add src/ to Python path."""

import sys
from pathlib import Path

# Ensure the src directory is on the import path
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
