# test_adder_tree.py
from   decimal import Decimal
import math
from   pathlib import Path
import pytest

from util.utilities import runner, lint, assert_resolvable, sign_extend
tbpath = Path(__file__).parent

import cocotb
from cocotb.triggers import Timer
   
import random
random.seed(50)

timescale = "1ps/1ps"

tests = ['single_test'
        ,'full_bw_test']

def pack_addends(addends, in_bits):
    packed = 0
    mask = (1 << in_bits) - 1
    for i, x in enumerate(addends):
        packed |= (x & mask) << (i * in_bits)
    return packed

def unpack_addends(packed, in_bits, addend_count):
    addends = []
    mask = (1 << in_bits) - 1
    for i in range(addend_count):
        raw = (packed >> (i * in_bits)) & mask
        addends.append(sign_extend(raw, in_bits))
    return addends

def acc_width(addend_count, in_width):
    return math.ceil(math.log2(addend_count)) + in_width

def trunc_signed(value: int, width: int) -> int:
    return sign_extend(value, width)

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
    [( 2,  2, output_width( 2,  2)),
     ( 2,  3, output_width( 2,  3)),
     ( 4,  5, output_width( 4,  5)),
     ( 8,  8, output_width( 8,  8)),
     ( 8, 16, output_width( 8, 16)),
     ( 8, 32, output_width( 8, 32)),
    ],
)

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
        self._dut = dut
        self._addends_i = dut.addends_i
        self._sum_o = dut.sum_o

        self._InBits      = int(dut.InBits.value)
        self._OutBits     = int(dut.OutBits.value)    
        self._AddendCount = int(dut.AddendCount.value)

    def consume(self):
        packed = int(self._addends_i.value.integer)
        addends = unpack_addends(packed, self._InBits, self._AddendCount)
        return trunc_signed(sum(addends), self._OutBits)

    def produce(self, expected):
        assert_resolvable(self._sum_o)
        got = sign_extend(int(self._sum_o.value.integer), self._OutBits)
        exp = int(expected)

        print(f"Expected: {exp}, Got: {got}, Addends: {unpack_addends(int(self._addends_i.value.integer), self._InBits, self._AddendCount)}")
        assert got == exp, (
            f"Mismatch. Expected {exp}, got {got}"
        )
    
async def comb_step(dut, model, addends):
    in_bits = int(dut.InBits.value)
    dut.addends_i.value = pack_addends(addends, in_bits)

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

    lo = -(1 << (in_bits - 1))
    hi = (1 << (in_bits - 1)) - 1

    for _ in range(10):
        addends = [random.randint(lo, hi) for _ in range(addend_count)]
        await comb_step(dut, model, addends)