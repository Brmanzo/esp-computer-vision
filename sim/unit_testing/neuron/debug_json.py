
import os
import json
from pathlib import Path

def test_debug():
    repo_root = Path("/home/brmanzo/cse121/esp/esp-computer-vision")
    tbpath = repo_root / "sim/unit_testing/neuron"
    
    use_dsp = True
    jsonname = str(repo_root / "filelists" / ("neuron_dsp.json" if use_dsp else "neuron.json"))
    
    print(f"DEBUG: use_dsp={use_dsp}")
    print(f"DEBUG: jsonname={jsonname}")
    
    with open(jsonname) as f:
        data = json.load(f)
        print(f"DEBUG: JSON content={data}")

if __name__ == "__main__":
    test_debug()
