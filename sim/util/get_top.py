import json, sys
from pathlib import Path

filelist_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("filelists/top.json")

with filelist_path.open() as f:
    print(json.load(f)["top"])