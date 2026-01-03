## DO NOT MODIFY ANYTHING IN THIS FILE WITHOUT PERMISSION FROM THE INSTRUCTOR OR TAs

# Path to the repository root
REPO_ROOT ?= $(shell git rev-parse --show-toplevel)

# If you have the tools installed in a non-standard path,
# you can override these to specify the path to the executable.
SBY ?= sby

.PHONY: check check-help check-clean check-vars-help

SBY_PATH ?= ../formal.sby
check: 
	$(SBY) -f $(SBY_PATH)

check-help:
	@echo "  check: Run the formal verification toolchain on the verilog module"

check-vars-help:
	@echo "    SBY: Override this variable to set the location of your yosys executable."

check-clean:
	rm -rf *.yslog
	rm -rf $(TOP_MODULE)
	rm -rf $(TOP_MODULE)_synth

clean: check-clean
targets-help: check-help
vars-help: check-vars-help

.PHONY: check check-help check-clean check-vars-help

