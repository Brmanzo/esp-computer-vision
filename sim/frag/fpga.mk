# Find repo root first
REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)

# Include user overrides from repo root (works no matter current directory)
-include $(REPO_ROOT)/config.mk

# If you have the tools installed in a non-standard path,
# you can override these to specify the path to the executable.
NEXTPNR ?= nextpnr-ice40
ICEPROG ?= iceprog
ICEPACK ?= icepack
ICETIME ?= icetime

# This is the default location for the icebreaker Pin Constraints File
# (PCF) Each part may use a different pcf file, so check in the partX
# directory first! Derived from
# https://github.com/icebreaker-fpga/icebreaker-verilog-examples/blob/main/icebreaker/icebreaker.pcf
PCF_PATH = $(REPO_ROOT)/boards/icebreakerV1_1a/icebreaker.pcf
prog: ice40.bin
	$(ICEPROG) $<

# Placement & Route. Depends on synth.mk
ice40.asc: ice40.json $(PCF_PATH)
	$(NEXTPNR) -ql ice40.nplog --up5k --package sg48 --freq 12 \
	  --asc $@ --pcf $(PCF_PATH) --json $< --top top \
	  --report ice40_report.json --no-promote-globals
	  

# Bitstream generation.
bitstream: ice40.bin
ice40.bin: ice40.asc
	$(ICEPACK) $< $@

buildflash: ice40.bin
	$(ICEPROG) ice40.bin

util: ice40.asc
	@awk '/Device utilisation:/{flag=1} flag{print} /ICESTORM_SPRAM/{exit}' ice40.nplog

gbuf: ice40.nplog
	@grep -niE "global|gbuf|sb_gb|promot|glb" ice40.nplog

.PHONY: util gbuf

# Timing analysis
ice40.rpt: ice40.asc
	$(ICETIME) -d up5k -c 12 -mtr $@ $<

fpga-clean:
	rm -rf ice40.bin
	rm -rf ice40.rpt
	rm -rf ice40.asc
	rm -rf ice40.nplog
	rm -rf ice40_report.json

fpga-help:
	@echo "  bitstream: Build the FPGA program (bitstream)"
	@echo "  prog: Flash the bistream to your FPGA (If running locally)"

fpga-vars-help:
	@echo "    NEXTPNR: Override this variable to set the location of your nextpnr executable."
	@echo "    ICEPROG: Override this variable to set the location of your Icebreaker Programmer executable."
	@echo "    ICEPACK: Override this variable to set the location of your icepack executable."
	@echo "    ICETIME: Override this variable to set the location of your icetime executable."

clean: fpga-clean
targets-help: fpga-help
vars-help: fpga-vars-help

.PHONY: prog fpga-help fpga-clean fpga-vars-help vars-help targets-help clean
