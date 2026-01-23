REPO_ROOT ?= $(shell git rev-parse --show-toplevel 2>/dev/null)

PCF_PATH = $(REPO_ROOT)/boards/icebreakerV1_1a/icebreaker.pcf
-include $(REPO_ROOT)/sim/frag/simulate.mk
-include $(REPO_ROOT)/sim/frag/synth.mk
-include $(REPO_ROOT)/sim/frag/fpga.mk

UNIT_TEST_DIRS := $(filter-out \
  $(REPO_ROOT)/sim/unit_testing/rle_encode \
  $(REPO_ROOT)/sim/unit_testing/rle_decode, \
  $(wildcard $(REPO_ROOT)/sim/unit_testing/*))

.PHONY: clean-all

clean-all: clean
	@for d in $(UNIT_TEST_DIRS); do \
	  if [ -f $$d/Makefile ]; then \
	    echo "Cleaning $${d#$(REPO_ROOT)/}"; \
	    $(MAKE) -C $$d clean; \
	  fi; \
	done

.PHONY: test-all

test-all:
	@set -e; \
	for d in $(UNIT_TEST_DIRS); do \
	  if [ -f $$d/Makefile ]; then \
	    echo "Testing $${d#$(REPO_ROOT)/}"; \
	    $(MAKE) -C $$d test; \
	  fi; \
	done