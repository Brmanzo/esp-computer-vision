# sim/conftest.py
import sys
import os
from pathlib import Path

# Get the repo root relative to this file
REPO_ROOT = Path(__file__).parent.parent

# Add paths once for the entire session
sys.path.insert(0, str(REPO_ROOT / "sim"))