# Find repo root first
REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)

# Include user overrides from repo root (works no matter current directory)
-include $(REPO_ROOT)/config.mk

YOSYS      ?= yosys
NETLISTSVG ?= netlistsvg
RSVG       ?= rsvg-convert

FILELIST ?= $(REPO_ROOT)/filelists/top.json
TOP_SV   ?= $(REPO_ROOT)/rtl/top/top.sv

SYNTH_SOURCES = $(shell python3 $(REPO_ROOT)/sim/util/get_filelist.py $(FILELIST))
ABSTRACT_TOP  = $(shell python3 $(REPO_ROOT)/sim/util/get_top.py $(FILELIST))

# The ice40 commands will only work if top.sv is provided, i.e. if
# there is a design for the FPGA.
ice40.pdf: ice40.json
	$(NETLISTSVG) $< -o $(subst pdf,svg,$@)
	$(RSVG) -f pdf $(subst pdf,svg,$@) -o $@

synth-ice40: ice40.json
ice40.json: $(TOP_SV) $(FILELIST) $(SYNTH_SOURCES)
	$(YOSYS) -ql ice40.yslog -p 'read_verilog -sv $(TOP_SV) $(SYNTH_SOURCES); hierarchy -top top; synth_ice40 -dsp -top top; delete t:$$scopeinfo; clean -purge; write_json $@'

# These commands will always work.
mapped.pdf: mapped.json
	$(NETLISTSVG) $< -o $(subst pdf,svg,$@)
	$(RSVG) -f pdf $(subst pdf,svg,$@) -o $@

synth-mapped: mapped.json
mapped.json: $(FILELIST) $(SYNTH_SOURCES)
	$(YOSYS) -ql mapped.yslog -p 'read_verilog -sv $(SYNTH_SOURCES); hierarchy -top $(ABSTRACT_TOP); synth_ice40 -dsp -top $(ABSTRACT_TOP); delete t:$$scopeinfo; clean -purge; write_json $@'

synth-xilinx: mapped.json
xilinx.json: $(FILELIST) $(SYNTH_SOURCES)
	$(YOSYS) -ql xilinx.yslog -p 'read_verilog -sv $(SYNTH_SOURCES); hierarchy -top $(ABSTRACT_TOP); synth_xilinx -top $(ABSTRACT_TOP); xilinx_dsp; delete t:$$scopeinfo; clean -purge; write_json $@'

xilinx.pdf: xilinx.json
	$(NETLISTSVG) $< -o $(subst pdf,svg,$@)
	$(RSVG) -f pdf $(subst pdf,svg,$@) -o $@

# These commands will always work.
abstract.pdf: abstract.json
	$(NETLISTSVG) $< -o $(subst pdf,svg,$@)
	$(RSVG) -f pdf $(subst pdf,svg,$@) -o $@

synth-abstract: abstract.json
abstract.json: $(FILELIST) $(SYNTH_SOURCES)
	$(YOSYS) -ql abstract.yslog -p 'read_verilog -sv $(SYNTH_SOURCES); hierarchy -top $(ABSTRACT_TOP); proc; opt; flatten; delete t:$$scopeinfo; clean -purge; write_json $@'
synth-clean:
	rm -rf ice40.json
	rm -rf ice40.yslog
	rm -rf ice40.svg ice40.pdf

	rm -rf mapped.json
	rm -rf mapped.yslog
	rm -rf mapped.svg mapped.pdf

	rm -rf xilinx.json
	rm -rf xilinx.yslog
	rm -rf xilinx.svg xilinx.pdf

	rm -rf abstract.json
	rm -rf abstract.yslog
	rm -rf abstract.svg abstract.pdf

synth-help:
	@echo "  synth-abstract: Synthesize the circuit for abstract boolean gates, without a top level"
	@echo "  abstract.pdf: Generate the .pdf file for the ICE40 circuit, without a top level"
	@echo "  synth-mapped: Synthesize the circuit for the ICE40 FPGA, without a top level"
	@echo "  mapped.pdf: Generate the .pdf file for the ICE40 circuit, without a top level"
	@echo "  xilinx.pdf: Generate the .pdf file for the Xilinx circuit, without a top level"
	@echo "  synth-ice40: Synthesize the circuit for the ICE40 FPGA, with a top level"
	@echo "  ice40.pdf: Generate the .pdf file for the ICE40 circuit, with a top level"


synth-vars-help:
	@echo "    YOSYS: Override this variable to set the location of your yosys executable."

clean: synth-clean
targets-help: synth-help
vars-help: synth-vars-help

.PHONY: synth-clean synth-help synth-vars-help vars-help targets-help clean synth-mapped synth-clean synth-abstract
