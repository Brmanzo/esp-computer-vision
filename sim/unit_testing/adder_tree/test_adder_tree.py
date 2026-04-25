# test_adder_tree.py
from   decimal import Decimal
import math
from   pathlib import Path
import pytest

from util.utilities import runner, lint, assert_resolvable
from util.bitwise import sign_extend, pack_terms, unpack_terms
from util.gen_inputs import gen_input_channels
tbpath = Path(__file__).parent

import cocotb
from cocotb.triggers import Timer
   
import random
random.seed(50)

timescale = "1ps/1ps"

tests = ['single_test'
        ,'full_bw_test']

def output_width(width_in: int, addend_count: int) -> str:
    '''Calculates proper output width for signed two's complement accumulations.'''
    max_mag = 1 << (width_in - 1)  # The absolute max for signed is the negative boundary (e.g., -128 for 8-bit)
    max_sum = addend_count * max_mag
    abs_bits = max_sum.bit_length()
    return str(abs_bits + 1)

# Test that binary tree can accomodate 
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits, AddendCount, OutBits",
    [( 2,  2, output_width( 2,  2)) # TODO: Test InBits=1
    ,( 2,  3, output_width( 2,  3))
    ,( 4,  5, output_width( 4,  5))
    ,( 8,  8, output_width( 8,  8))
    ,( 8, 16, output_width( 8, 16))
    ,( 8, 32, output_width( 8, 32))
    ])

def test_each(test_name, simulator, InBits, OutBits, AddendCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_adder_tree")

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, AddendCount, OutBits", 
                         [(1, 2, output_width(1, 2)), (2, 3, output_width(2, 3))])
def test_lint(simulator, InBits, OutBits, AddendCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, AddendCount, OutBits", 
                         [(1, 2, output_width(1, 2)), (2, 3, output_width(2, 3))])
def test_style(simulator, InBits, OutBits, AddendCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

class AdderTreeModel():
    def __init__(self, dut):
        self._addends_i = dut.addends_i
        self._sum_o     = dut.sum_o

        self._InBits      = int(dut.InBits.value)
        self._OutBits     = int(dut.OutBits.value)    
        self._AddendCount = int(dut.AddendCount.value)

    def consume(self):
        packed = int(self._addends_i.value.integer)
        if self._InBits == 1:
            raw_bits = unpack_terms(packed, self._InBits, self._AddendCount)
            addends = [1 if bit == 1 else -1 for bit in raw_bits]
        else:   
            addends = unpack_terms(packed, self._InBits, self._AddendCount)

        return sign_extend(sum(addends), self._OutBits)

    def produce(self, expected):
        assert_resolvable(self._sum_o)
        got = sign_extend(int(self._sum_o.value.integer), self._OutBits)
        exp = int(expected)

        print(f"Expected: {exp}, Got: {got}, Addends: {unpack_terms(int(self._addends_i.value.integer), self._InBits, self._AddendCount)}")
        assert got == exp, (
            f"Mismatch. Expected {exp}, got {got}"
        )
    
async def comb_step(dut, model, addends):
    in_bits = int(dut.InBits.value)
    dut.addends_i.value = pack_terms(addends, in_bits)

    await Timer(Decimal(1), units="step")

    expected = model.consume()
    model.produce(expected)

@cocotb.test
async def single_test(dut):
    addend_count = int(dut.AddendCount.value)
    model = AdderTreeModel(dut)

    addends = [1] * addend_count
    await comb_step(dut, model, addends)

@cocotb.test
async def full_bw_test(dut):
    in_bits = int(dut.InBits.value)
    addend_count = int(dut.AddendCount.value)
    model = AdderTreeModel(dut)

    for _ in range(10):
        await comb_step(dut, model, gen_input_channels(in_bits, addend_count))