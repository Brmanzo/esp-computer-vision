REPO_ROOT ?= $(shell git rev-parse --show-toplevel 2>/dev/null)

PCF_PATH = $(REPO_ROOT)/boards/icebreakerV1_1a/icebreaker.pcf
-include $(REPO_ROOT)/sim/frag/simulate.mk
-include $(REPO_ROOT)/sim/frag/synth.mk
-include $(REPO_ROOT)/sim/frag/fpga.mk

UNIT_TEST_DIRS := $(sort $(dir $(wildcard $(REPO_ROOT)/sim/unit_testing/*/Makefile)))

.PHONY: clean-all
clean-all: clean
	@set -e; \
	for d in $(UNIT_TEST_DIRS); do \
	  echo "Cleaning $${d#$(REPO_ROOT)/}"; \
	  $(MAKE) -C "$$d" clean; \
	done

.PHONY: test-all
test-all:
	@set -e; \
	for d in $(UNIT_TEST_DIRS); do \
	  echo "Testing $${d#$(REPO_ROOT)/}"; \
	  $(MAKE) -C "$$d" test VERBOSE=0; \
	done

lint-all:
	@set -e; \
	for d in $(UNIT_TEST_DIRS); do \
	  echo "Linting $${d#$(REPO_ROOT)/}"; \
	  $(MAKE) -C "$$d" test lint VERBOSE=0; \
	  $(MAKE) -C "$$d" lint; \
	done

.PHONY: test-one
test-one:
	@if [ -z "$(DIR)" ]; then \
	  echo "Usage: make test-one DIR=sim/unit_testing/<name>"; \
	  exit 2; \
	fi
	@echo "Testing $(DIR)"; \
	$(MAKE) -C "$(DIR)" test