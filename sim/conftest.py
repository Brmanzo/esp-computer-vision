# sim/conftest.py
import sys
import os
import pytest
from pathlib import Path

# Get the repo root relative to this file
REPO_ROOT = Path(__file__).parent.parent

# Add paths once for the entire session
sys.path.insert(0, str(REPO_ROOT / "sim"))

def pytest_addoption(parser):
    parser.addoption("--dsp", action="store_true", default=False, help="Test using neuron_dsp implementation")

@pytest.fixture
def use_dsp(request):
    return request.config.getoption("dsp")