# sim/unit_testing/Makefile
# Run clean targets for every unit test directory that has a Makefile.

REPO_ROOT ?= $(shell git rev-parse --show-toplevel)

# Find immediate subdirectories containing a Makefile
TEST_DIRS := $(sort $(dir $(wildcard */Makefile)))
TEST_DIRS := $(patsubst %/,%,$(TEST_DIRS))

.PHONY: help list clean synth-clean fpga-clean all-clean \
        clean-% synth-clean-% fpga-clean-%

help:
	@echo "Targets:"
	@echo "  list          - list unit-test dirs detected"
	@echo "  clean         - run 'make clean' in each dir"
	@echo "  synth-clean   - run 'make synth-clean' in each dir"
	@echo "  fpga-clean    - run 'make fpga-clean' in each dir (if exists)"
	@echo "  all-clean     - run clean + synth-clean + fpga-clean"

list:
	@echo $(TEST_DIRS)

# Generic pattern to run a target in one directory
run-%:
	@set -e; \
	for d in $(TEST_DIRS); do \
		echo "==> $$d: make $*"; \
		$(MAKE) -C $$d $*; \
	done

clean: run-clean
synth-clean: run-synth-clean
fpga-clean: run-fpga-clean

all-clean:
	@$(MAKE) clean
	@$(MAKE) synth-clean
	@$(MAKE) fpga-clean

# Optional: single-directory convenience
clean-%:
	@$(MAKE) -C $* clean
synth-clean-%:
	@$(MAKE) -C $* synth-clean
fpga-clean-%:
	@$(MAKE) -C $* fpga-clean