REPO_ROOT ?= $(shell git rev-parse --show-toplevel)

PCF_PATH = $(REPO_ROOT)/boards/icebreakerV1_1a/icebreaker.pcf
DSP ?= 0
ifeq ($(DSP),1)
FILELIST = filter_dsp.json
else
FILELIST = filter.json
endif
export FILELIST
-include $(REPO_ROOT)/sim/frag/simulate.mk
-include $(REPO_ROOT)/sim/frag/synth.mk
-include $(REPO_ROOT)/sim/frag/fpga.mk
