#!/usr/bin/env python3
"""Dispatcher: cnn.py <tool> [args...]

Tools:
  bram   Verify BRAM contents against hex ROMs
  export        Quantize and export weights to hex files
  verilog       Render CNN architecture to SystemVerilog
  train         Train the neural network
  inference     Run inference
  sample        Sample/collect data
  arch          Architecture Report
"""
import sys
import runpy

TOOLS = {"bram", "export", "verilog", "train", "inference", "sample", "arch", "fpga"}

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    sys.exit(0)

tool = sys.argv[1]
if tool not in TOOLS:
    print(f"Unknown tool '{tool}'. Available: {', '.join(sorted(TOOLS))}")
    sys.exit(1)

sys.argv = sys.argv[1:]  # shift so the tool sees its own argv[0]
sys.argv[0] = f"nn.{tool}"
runpy.run_module(f"nn.{tool}", run_name="__main__", alter_sys=True)
