# Find repo root first
REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)

# Include user overrides from repo root (works no matter current directory)
-include $(REPO_ROOT)/config.mk
export PYTHONPATH := $(REPO_ROOT)/sim/util:$(PYTHONPATH)

# Default tool locations (can be overridden in environment or config.mk)
IVERILOG  ?= iverilog
VERILATOR ?= verilator
VERIBLE   ?= verible-verilog-lint
PYTHON3   ?= python

VERIBLE_EXCLUDES := %/rtl/vendor/%
VERIBLE_SOURCES := $(filter-out $(VERIBLE_EXCLUDES),$(LINT_SOURCES))

# Ensure user-local bins are visible when running under make
export PATH := $(HOME)/.local/bin:$(PATH)

FILELIST ?= $(REPO_ROOT)/filelists/top.json
SIM_SOURCES = $(shell $(PYTHON3) $(REPO_ROOT)/sim/util/get_filelist.py $(FILELIST))
SIM_SOURCES := $(foreach f,$(SIM_SOURCES),$(if $(filter /%,$(f)),$(f),$(REPO_ROOT)/$(f)))
SIM_TOP     = $(shell $(PYTHON3) $(REPO_ROOT)/sim/util/get_top.py $(FILELIST))

FILE_ABS := $(if $(FILE),$(if $(filter /%,$(FILE)),$(FILE),$(REPO_ROOT)/$(FILE)),)

LINT_SOURCES := $(if $(FILE_ABS),$(FILE_ABS),$(SIM_SOURCES))

all: help

ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
# Run both simulators
test: results.json

results.json: $(FILELIST) $(SIM_SOURCES)
	$(PYTHON3) -m pytest -rA $(if $(ARGS),-k "$(ARGS)",)

# lint runs the Verilator linter on your code.
lint: lint-verilator lint-verible

lint-verilator:
	@command -v $(VERILATOR) >/dev/null || { \
		echo "ERROR: $(VERILATOR) not found in PATH"; exit 1; }
	$(VERILATOR) --lint-only -Wall \
	  $(if $(FILE_ABS),,$(if $(SIM_TOP),-top $(SIM_TOP),)) \
	  $(LINT_SOURCES)

lint-verible:
	@command -v $(VERIBLE) >/dev/null || { \
		echo "ERROR: $(VERIBLE) not found in PATH"; exit 1; }
	$(VERIBLE) --lint_fatal --parse_fatal $(VERIBLE_SOURCES)

# Remove all compiler outputs
sim-clean:
	rm -rf run
	rm -rf build
	rm -rf lint
	rm -rf __pycache__
	rm -rf .pytest_cache

# Remove all generated files
extraclean: clean
	rm -f results.json
	rm -f verilator.json
	rm -f icarus.json

sim-help:
	@echo "  test: Shortcut for results.json"
	@echo "  results.json: Run all simulation tests"
	@echo "  lint: Run the Verilator linter on all source files"
	@echo "  clean: Remove all compiler outputs."
	@echo "  extraclean: Remove all generated files (runs clean)"

vars-intro-help:
	@echo ""
	@echo "  Optional Environment Variables:"

sim-vars-help:
	@echo "    VERILATOR: Override this variable to set the location of your verilator executable."
	@echo "    IVERILOG: Override this variable to set the location of your iverilog executable."

clean: sim-clean
targets-help: sim-help
vars-help: vars-intro-help sim-vars-help

help: targets-help vars-help 

print-config:
	@echo "REPO_ROOT=$(REPO_ROOT)"
	@echo "config.mk included? (check by printing YOSYS=$(YOSYS))"
	@echo "YOSYS=$(YOSYS)"

.PHONY: all test lint lint-verilator lint-verible sim-clean extraclean sim-help vars-intro-help sim-vars-help clean targets-help vars-help help test results.json
