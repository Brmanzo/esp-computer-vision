# Find repo root first
REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)

# Include user overrides from repo root (works no matter current directory)
-include $(REPO_ROOT)/config.mk

YOSYS      ?= yosys
NETLISTSVG ?= netlistsvg
RSVG       ?= rsvg-convert

FILELIST ?= $(REPO_ROOT)/filelists/top.json
TOP_SV   ?= $(REPO_ROOT)/rtl/top/top.sv
RTL_ROOT ?= $(REPO_ROOT)/rtl

SYNTH_SOURCES = $(shell python3 $(REPO_ROOT)/sim/util/get_filelist.py $(FILELIST))
ABSTRACT_TOP  = $(shell python3 $(REPO_ROOT)/sim/util/get_top.py $(FILELIST))

define find_sv
	$(firstword $(wildcard $(RTL_ROOT)/**/$1.sv))
endef

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

%.json:
	@sv="$(call find_sv,$*)"; \
	if [ -z "$$sv" ]; then \
	  echo "ERROR: Could not find SV file for module '$*' under $(RTL_ROOT)/**/$*.sv"; \
	  exit 1; \
	fi; \
	echo "[Yosys] $$sv -> $@ (top=$*)"; \
	$(YOSYS) -ql $*.yslog -p "read_verilog -sv $(SYNTH_SOURCES); hierarchy -top $*; proc; opt; delete t:\$$scopeinfo; clean -purge; write_json $@"

%.pdf: %.json
	$(NETLISTSVG) $< -o $(subst pdf,svg,$@)
	$(RSVG) -f pdf $(subst pdf,svg,$@) -o $@

%.mapped.json:
	@sv="$(call find_sv,$*)"; \
	if [ -z "$$sv" ]; then \
	  echo "ERROR: Could not find SV file for module '$*' under $(RTL_ROOT)/**/$*.sv"; \
	  exit 1; \
	fi; \
	echo "[Yosys] $$sv -> $@ (synth_ice40 top=$*)"; \
	$(YOSYS) -ql $*.mapped.yslog -p "read_verilog -sv $(SYNTH_SOURCES); hierarchy -top $*; synth_ice40 -dsp -top $*; delete t:\$$scopeinfo; clean -purge; write_json $@"

%.mapped.pdf: %.mapped.json
	$(NETLISTSVG) $< -o $(subst .pdf,.svg,$@)
	$(RSVG) -f pdf $(subst .pdf,.svg,$@) -o $@

.PHONY: module-help
module-help:
	@echo "Per-module targets:"
	@echo "  make <mod>.pdf         (abstract proc/opt netlist for <mod>, found under rtl/**/<mod>.sv)"
	@echo "  make <mod>.mapped.pdf  (iCE40 mapped netlist for <mod>)"

synth-clean:
	rm -rf ice40.json

	rm -rf mapped.json

	rm -rf xilinx.json

	rm -rf abstract.json
	rm -rf abstract.yslog
	rm -rf *.svg *.pdf *.yslog

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
