# test_comparator_tree.py
from   decimal import Decimal
import pytest
from   pathlib import Path

from util.utilities import runner, lint, assert_resolvable
from util.bitwise   import sign_extend, pack_terms, unpack_terms
from util.gen_inputs import gen_input_channels
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import Timer
   
import random
random.seed(50)

timescale = "1ps/1ps"

tests = ['single_test'
        ,'full_bw_test']

# Test that binary tree can accomodate 
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits, ClassCount",
    [( 2,  2), # TODO: Test InBits=1
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
        classes = unpack_terms(packed, self._InBits, self._ClassCount)
        return (sign_extend(max(classes), self._OutBits), classes.index(max(classes)))

    def produce(self, expected):
        assert_resolvable(self._max_o)
        got_max = sign_extend(int(self._max_o.value.integer), self._OutBits)
        exp_max = int(expected[0])

        got_id = int(self._id_o.value.integer)
        exp_id = int(expected[1])

        print(f"Expected: {exp_max}, Got: {got_max}, Classes: {unpack_terms(int(self._classes_i.value.integer), self._InBits, self._ClassCount)}")
        print(f"Expected ID: {exp_id}, Got ID: {int(self._id_o.value.integer)}")
        assert got_max == exp_max, (
            f"Mismatch. Exp_maxected {exp_max}, got {got_max}"
        )
        assert got_id == exp_id, (
            f"ID Mismatch. Expected {exp_id}, got {got_id}"
        )   
    
async def comb_step(dut, model, classes):
    in_bits = int(dut.InBits.value)
    dut.classes_i.value = pack_terms(classes, in_bits)

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
    
    for _ in range(10):
        await comb_step(dut, model, gen_input_channels(in_bits, class_count))