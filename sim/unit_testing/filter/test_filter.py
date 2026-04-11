import os
import sys
import git
import queue
import math
import numpy as np
from typing import List
from decimal import Decimal
import torch
import torch.nn as nn
import shutil

# I don't like this, but it's convenient.
_REPO_ROOT = git.Repo(search_parent_directories=True).working_tree_dir
assert _REPO_ROOT is not None, "REPO_ROOT path must not be None"
assert (os.path.exists(_REPO_ROOT)), "REPO_ROOT path must exist"
sys.path.append(os.path.join(_REPO_ROOT, "util"))
from utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence, delay_cycles, ReadyValidInterface
tbpath = os.path.dirname(os.path.realpath(__file__))

import pytest

import cocotb

from cocotb.utils import get_sim_time
from cocotb.triggers import Timer, RisingEdge, FallingEdge, with_timeout
from cocotb.result import SimTimeoutError
   
import random
random.seed(50)

timescale = "1ps/1ps"

tests = ['single_test'
        ,'full_bw_test']

def sign_extend(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    sign_bit = 1 << (width - 1)
    return (value ^ sign_bit) - sign_bit

def pack_inputs(values, bits, in_channels, term_count):
    packed = 0
    mask = (1 << bits) - 1
    for ch in range(in_channels):
        for i in range(term_count):
            packed |= (values[ch][i] & mask) << ((ch * term_count + i) * bits)
    return packed

def unpack_inputs(packed, in_bits, in_channels, term_count):
    terms = [[] for _ in range(in_channels)]
    mask = (1 << in_bits) - 1
    for ch in range(in_channels):
        for i in range(term_count):
            raw = (packed >> ((ch * term_count + i) * in_bits)) & mask
            terms[ch].append(sign_extend(raw, in_bits))
    return terms    

def unpack_unsigned_inputs(packed, in_bits, in_channels, term_count):
    terms = [[] for _ in range(in_channels)]
    mask = (1 << in_bits) - 1
    for ch in range(in_channels):
        for i in range(term_count):
            raw = (packed >> ((ch * term_count + i) * in_bits)) & mask
            terms[ch].append(raw) # NO sign extension!
    return terms    

def trunc_signed(value: int, width: int) -> int:
    return sign_extend(value, width)

def acc_width(in_bits: int, weight_bits: int, kernel_width: int, in_channels: int) -> int:
    kernel_area = kernel_width * kernel_width

    if in_bits == 1:
        max_input_mag = 1      # bipolar {-1, +1}
    else:
        max_input_mag = (1 << in_bits) - 1

    max_weight_mag = 1 << (weight_bits - 1)   # e.g. 2-bit signed => magnitude 2
    max_sum_mag = kernel_area * in_channels * max_input_mag * max_weight_mag

    return max_sum_mag.bit_length() + 1

# Test that binary tree can accomodate 
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits, WeightBits, KernelWidth, InChannels, AccBits, OutBits",
    [( 1, 2, 3, 1, acc_width( 1, 2, 3,  1), 1),
     ( 1, 2, 3, 2, acc_width( 1, 2, 3,  2), acc_width( 1, 2, 3,  2)),
     ( 1, 3, 3, 1, acc_width( 1, 3, 3,  1), 1),
     ( 8, 2, 3, 1, acc_width( 8, 2, 3,  1), acc_width( 8, 2, 3,  1)),
     ( 8, 2, 3, 2, acc_width( 8, 2, 3,  2), 1),
     ( 8, 2, 3,16, acc_width( 8, 2, 3, 16), 1),
     ( 8, 2, 3,32, acc_width( 8, 2, 3, 32), acc_width( 8, 2, 3, 32)),
    ],
)

def test_each(test_name, simulator, InBits, WeightBits, KernelWidth, InChannels, AccBits, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_filter")

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, KernelWidth, InChannels, AccBits, OutBits", 
                         [( 1, 2, 3, 1, acc_width( 1, 2, 3,  1), acc_width( 1, 2, 3,  1))])
def test_lint(simulator, InBits, WeightBits, KernelWidth, InChannels, AccBits, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, KernelWidth, InChannels, AccBits, OutBits", 
                         [( 1, 2, 3, 1, acc_width( 1, 2, 3,  1), acc_width( 1, 2, 3,  1))])
def test_style(simulator, InBits, WeightBits, KernelWidth, InChannels, AccBits, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

class FilterModel:
    def __init__(self, dut):
        self._dut = dut
        self._data_o = dut.data_o

        self._InBits      = int(dut.InBits.value)
        self._OutBits     = int(dut.OutBits.value)
        self._WeightBits  = int(dut.WeightBits.value)
        self._AccBits     = int(dut.AccBits.value)
        self._InChannels  = int(dut.InChannels.value)
        self._KernelWidth = int(dut.KernelWidth.value)
        self._KernelArea  = self._KernelWidth * self._KernelWidth

        self._weights = None
        self._windows = None

    def consume(self, windows, weights):
        self._windows = windows
        self._weights = weights

        total = 0
        for ch in range(self._InChannels):
            for i in range(self._KernelArea):
                win = windows[ch][i]
                wgt = weights[ch][i]
                # If input is 1-bit, treat as bipolar {-1, +1} instead of unsigned {0, 1}.
                if self._InBits == 1:
                    win = 1 if win == 1 else -1

                total += win * wgt

        if self._OutBits == 1:
            return 1 if total > 0 else 0
        else:
            return trunc_signed(total, self._OutBits)

    def produce(self, expected):
        assert_resolvable(self._data_o)

        # If output is single-bit, treat as unsigned 0/1 based on sign of total.
        if self._OutBits == 1:
            got = int(self._data_o.value.integer) & 1
        # For multi-bit outputs, we need to sign-extend the value from the DUT before comparing.
        else:
            got = sign_extend(int(self._data_o.value.integer), self._OutBits)

        print(f"Expected: {expected}, Got: {got}, windows: {self._windows}, weights: {self._weights}")
        assert got == expected, f"Mismatch. Expected {expected}, got {got}"
    
async def comb_step(dut, model, windows, weights):
    in_bits     = int(dut.InBits.value)
    weight_bits = int(dut.WeightBits.value)
    in_channels = int(dut.InChannels.value)
    term_count  = int(dut.KernelWidth.value) * int(dut.KernelWidth.value)

    dut.windows_i.value = pack_inputs(windows, in_bits, in_channels, term_count)
    dut.weights_i.value = pack_inputs(weights, weight_bits, in_channels, term_count)

    await Timer(Decimal(1), units="step")

    expected = model.consume(windows, weights)
    model.produce(expected)


@cocotb.test
async def single_test(dut):
    term_count = int(dut.KernelWidth.value) * int(dut.KernelWidth.value)
    in_channels = int(dut.InChannels.value)

    model = FilterModel(dut)

    windows = [[1] * term_count for _ in range(in_channels)]
    weights = [[1] * term_count for _ in range(in_channels)]
    await comb_step(dut, model, windows, weights)


@cocotb.test
async def full_bw_test(dut):
    in_bits     = int(dut.InBits.value)
    weight_bits = int(dut.WeightBits.value)
    in_channels = int(dut.InChannels.value)
    term_count  = int(dut.KernelWidth.value) * int(dut.KernelWidth.value)
    model       = FilterModel(dut)

    # Window is unsigned (0 to Max)
    window_lo = 0
    window_hi = (1 << in_bits) - 1

    # Weights are signed two's complement (Min to Max)
    weight_hi =  (1 << (weight_bits - 1)) - 1
    weight_lo = -(1 << (weight_bits - 1))

    for _ in range(10): # Note: Feel free to bump this up to 500 now!
        windows = [[random.randint(window_lo, window_hi) for _ in range(term_count)] for _ in range(in_channels)]
        weights = [[random.randint(weight_lo, weight_hi) for _ in range(term_count)] for _ in range(in_channels)]
        await comb_step(dut, model, windows, weights)