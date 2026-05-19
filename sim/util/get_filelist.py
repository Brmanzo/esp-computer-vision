import json, sys
from pathlib import Path

filelist_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rtl/top/top.json")

with filelist_path.open() as f:
    files = json.load(f)["files"]

print(" ".join(files))