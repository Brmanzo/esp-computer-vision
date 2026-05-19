#!/usr/bin/env python3
"""Dispatcher: cnn.py <tool> [args...]

Tools:
  bram_decode   Verify BRAM contents against hex ROMs
  export        Quantize and export weights to hex files
  render        Render CNN architecture to SystemVerilog
  train         Train the model
  inference     Run inference
  sample        Sample/collect data
  model         Model utilities
"""
import sys
import runpy

TOOLS = {"bram_decode", "export", "render", "train", "inference", "sample", "model"}

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    sys.exit(0)

tool = sys.argv[1]
if tool not in TOOLS:
    print(f"Unknown tool '{tool}'. Available: {', '.join(sorted(TOOLS))}")
    sys.exit(1)

sys.argv = sys.argv[1:]  # shift so the tool sees its own argv[0]
sys.argv[0] = f"model.{tool}"
runpy.run_module(f"model.{tool}", run_name="__main__", alter_sys=True)
