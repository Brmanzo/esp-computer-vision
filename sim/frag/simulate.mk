# Path to the repository root
REPO_ROOT ?= $(shell git rev-parse --show-toplevel)

# If you have iverilog or verilator installed in a non-standard path,
# you can override these to specify the path to the executable.
IVERILOG ?= iverilog
VERILATOR ?= verilator

# This is a little bit hacky, but sufficient. In order to make sure
# that students can edit the filelist, that make knows about updates
# to that filelist *and* the files themselves, and that pytest can
# read the filelist, we store the listlist in a json file. We then
# read the json file while checking dependencies.
FILELIST ?= $(REPO_ROOT)/filelists/top.json
SIM_SOURCES = $(shell python3 $(REPO_ROOT)/sim/util/get_filelist.py $(FILELIST))
SIM_SOURCES := $(addprefix $(REPO_ROOT)/,$(SIM_SOURCES))
SIM_TOP     = $(shell python3 $(REPO_ROOT)/sim/util/get_top.py $(FILELIST))

all: help

# Run both simulators
test: results.json

results.json: $(FILELIST) $(SIM_SOURCES)
	pytest -rA

# lint runs the Verilator linter on your code.
lint:
	$(VERILATOR) --lint-only -top $(SIM_TOP) $(SIM_SOURCES) -Wall

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

.PHONY: all test lint sim-clean extraclean sim-help vars-intro-help sim-vars-help clean targets-help vars-help help test results.json
