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

_REPO_ROOT = git.Repo(search_parent_directories=True).working_tree_dir
assert _REPO_ROOT is not None, "REPO_ROOT path must not be None"
assert (os.path.exists(_REPO_ROOT)), "REPO_ROOT path must exist"
_UTIL_PATH = os.path.join(_REPO_ROOT, "sim", "util")
assert os.path.exists(_UTIL_PATH), f"Utilities path does not exist: {_UTIL_PATH}"
sys.path.insert(0, _UTIL_PATH)
from utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence, delay_cycles
tbpath = os.path.dirname(os.path.realpath(__file__))

import pytest

import cocotb

from cocotb.triggers import Timer

from cocotb_test.simulator import run
   
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

def pack_classes(classes, in_bits):
    packed = 0
    mask = (1 << in_bits) - 1
    for i, x in enumerate(classes):
        packed |= (x & mask) << (i * in_bits)
    return packed

def unpack_classes(packed, in_bits, class_count):
    classes = []
    mask = (1 << in_bits) - 1
    for i in range(class_count):
        raw = (packed >> (i * in_bits)) & mask
        classes.append(sign_extend(raw, in_bits))
    return classes

def acc_width(class_count, in_width):
    return math.ceil(math.log2(class_count)) + in_width

def trunc_signed(value: int, width: int) -> int:
    return sign_extend(value, width)

# Test that binary tree can accomodate 
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits, ClassCount",
    [( 2,  2),
     ( 2,  3),
     ( 4,  5),
     ( 8,  8),
     ( 8, 16),
     ( 8, 32),
    ],
)

def test_each(test_name, simulator, InBits, ClassCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_comparator_tree")

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, ClassCount", 
                         [(1, 2), (2, 3)])
def test_lint(simulator, InBits, ClassCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, ClassCount", 
                         [(1, 2), (2, 3)])
def test_style(simulator, InBits, ClassCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

class ComparatorTreeModel():
    def __init__(self, dut):
        self._dut = dut
        self._classes_i = dut.classes_i
        self._max_o = dut.max_o
        self._id_o = dut.id_o

        self._InBits      = int(dut.InBits.value)
        self._OutBits     = int(dut.OutBits.value)
        self._IdBits      = int(dut.IdBits.value)
        self._ClassCount = int(dut.ClassCount.value)

    def consume(self):
        packed = int(self._classes_i.value.integer)
        classes = unpack_classes(packed, self._InBits, self._ClassCount)
        return (trunc_signed(max(classes), self._OutBits), classes.index(max(classes)))

    def produce(self, expected):
        assert_resolvable(self._max_o)
        got_max = sign_extend(int(self._max_o.value.integer), self._OutBits)
        exp_max = int(expected[0])

        got_id = int(self._id_o.value.integer)
        exp_id = int(expected[1])

        print(f"Expected: {exp_max}, Got: {got_max}, Classes: {unpack_classes(int(self._classes_i.value.integer), self._InBits, self._ClassCount)}")
        print(f"Expected ID: {exp_id}, Got ID: {int(self._id_o.value.integer)}")
        assert got_max == exp_max, (
            f"Mismatch. Exp_maxected {exp_max}, got {got_max}"
        )
        assert got_id == exp_id, (
            f"ID Mismatch. Expected {exp_id}, got {got_id}"
        )   
    
async def comb_step(dut, model, classes):
    in_bits = int(dut.InBits.value)
    dut.classes_i.value = pack_classes(classes, in_bits)

    await Timer(Decimal(1), units="step")

    expected = model.consume()
    model.produce(expected)


@cocotb.test
async def single_test(dut):
    class_count = int(dut.ClassCount.value)
    model = ComparatorTreeModel(dut)

    classes = [1] * class_count
    await comb_step(dut, model, classes)


@cocotb.test
async def full_bw_test(dut):
    in_bits = int(dut.InBits.value)
    class_count = int(dut.ClassCount.value)
    model = ComparatorTreeModel(dut)

    lo = -(1 << (in_bits - 1))
    hi = (1 << (in_bits - 1)) - 1

    for _ in range(10):
        classes = [random.randint(lo, hi) for _ in range(class_count)]
        await comb_step(dut, model, classes)