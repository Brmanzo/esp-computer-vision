REPO_ROOT ?= $(shell git rev-parse --show-toplevel)

PCF_PATH = $(REPO_ROOT)/boards/icebreakerV1_1a/icebreaker.pcf
FILELIST = $(REPO_ROOT)/filelists/unpacker.json
export FILELIST
-include $(REPO_ROOT)/frag/simulate.mk
-include $(REPO_ROOT)/frag/synth.mk
-include $(REPO_ROOT)/frag/fpga.mk

