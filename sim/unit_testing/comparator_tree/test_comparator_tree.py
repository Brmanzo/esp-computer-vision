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

def pack_terms(terms, in_bits):
    packed = 0
    mask = (1 << in_bits) - 1
    for i, x in enumerate(terms):
        packed |= (x & mask) << (i * in_bits)
    return packed

def unpack_terms(packed, in_bits, term_count):
    terms = []
    mask = (1 << in_bits) - 1
    for i in range(term_count):
        raw = (packed >> (i * in_bits)) & mask
        terms.append(sign_extend(raw, in_bits))
    return terms

def acc_width(term_count, in_width):
    return math.ceil(math.log2(term_count)) + in_width

def trunc_signed(value: int, width: int) -> int:
    return sign_extend(value, width)

# Test that binary tree can accomodate 
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits, TermCount",
    [( 2,  2),
     ( 2,  3),
     ( 4,  5),
     ( 8,  8),
     ( 8, 16),
     ( 8, 32),
    ],
)

def test_each(test_name, simulator, InBits, TermCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_comparator_tree")

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, TermCount", 
                         [(1, 2), (2, 3)])
def test_lint(simulator, InBits, TermCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, TermCount", 
                         [(1, 2), (2, 3)])
def test_style(simulator, InBits, TermCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

class ComparatorTreeModel():
    def __init__(self, dut):
        self._dut = dut
        self._terms_i = dut.terms_i
        self._max_o = dut.max_o
        self._id_o = dut.id_o

        self._InBits      = int(dut.InBits.value)
        self._OutBits     = int(dut.OutBits.value)
        self._IdBits      = int(dut.IdBits.value)
        self._TermCount = int(dut.TermCount.value)

    def consume(self):
        packed = int(self._terms_i.value.integer)
        terms = unpack_terms(packed, self._InBits, self._TermCount)
        return (trunc_signed(max(terms), self._OutBits), terms.index(max(terms)))

    def produce(self, expected):
        assert_resolvable(self._max_o)
        got_max = sign_extend(int(self._max_o.value.integer), self._OutBits)
        exp_max = int(expected[0])

        got_id = int(self._id_o.value.integer)
        exp_id = int(expected[1])

        print(f"Expected: {exp_max}, Got: {got_max}, Terms: {unpack_terms(int(self._terms_i.value.integer), self._InBits, self._TermCount)}")
        print(f"Expected ID: {exp_id}, Got ID: {int(self._id_o.value.integer)}")
        assert got_max == exp_max, (
            f"Mismatch. Exp_maxected {exp_max}, got {got_max}"
        )
        assert got_id == exp_id, (
            f"ID Mismatch. Expected {exp_id}, got {got_id}"
        )   
    
async def comb_step(dut, model, terms):
    in_bits = int(dut.InBits.value)
    dut.terms_i.value = pack_terms(terms, in_bits)

    await Timer(Decimal(1), units="step")

    expected = model.consume()
    model.produce(expected)


@cocotb.test
async def single_test(dut):
    term_count = int(dut.TermCount.value)
    model = ComparatorTreeModel(dut)

    terms = [1] * term_count
    await comb_step(dut, model, terms)


@cocotb.test
async def full_bw_test(dut):
    in_bits = int(dut.InBits.value)
    term_count = int(dut.TermCount.value)
    model = ComparatorTreeModel(dut)

    lo = -(1 << (in_bits - 1))
    hi = (1 << (in_bits - 1)) - 1

    for _ in range(10):
        terms = [random.randint(lo, hi) for _ in range(term_count)]
        await comb_step(dut, model, terms)