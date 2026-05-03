# test_mac.py
import os
from   pathlib import Path
import pytest

from util.utilities import runner, lint, assert_resolvable, load_tests_from_csv, \
                           sim_verbose, auto_unpack
from util.bitwise   import sign_extend, pack_terms, unpack_terms
from util.gen_inputs import gen_input_channels
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import Timer, Decimal
   
import random
random.seed(50)

timescale = "1ps/1ps"

tests = ['single_test'
        ,'full_bw_test']

def output_width(in_width: int, weight_width: int, term_count: int) -> str:
    '''Calculates proper output width for mixed signed/unsigned MAC.'''
    max_in = 1 if in_width == 1 else (1 << in_width) - 1 
    max_weight = 1 << (weight_width - 1) 
    max_sum = term_count * max_in * max_weight
    abs_bits = max_sum.bit_length()
    return str(abs_bits + 1)

auto_rules = [
    ("OutBits", "OutBits", lambda InBits, WeightBits, TermCount: output_width(InBits, WeightBits, TermCount))
]

TEST_CASES = load_tests_from_csv(os.path.join(tbpath, "test_cases.csv"), auto_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)

def test_each(test_name, simulator, InBits, WeightBits, TermCount, OutBits):
    # This line must be first
    parameters = dict(locals())
    parameters.pop('test_name', None)
    parameters.pop('simulator', None)
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_mac")

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, TermCount, OutBits", 
                         [(1, 2, 2, output_width(1, 2, 2))])
def test_lint(simulator, InBits, WeightBits, TermCount, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, TermCount, OutBits", 
                         [(1, 2, 2, output_width(1, 2, 2))])
def test_style(simulator, InBits, WeightBits, TermCount, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

class MacModel():
    def __init__(self, dut):
        self._dut = dut
        self._weights_i = dut.weights_i
        self._window_i = dut.window_i
        self._sum_o = dut.sum_o

        self._InBits      = int(dut.InBits.value)
        self._OutBits     = int(dut.OutBits.value)
        self._WeightBits  = int(dut.WeightBits.value)    
        self._TermCount   = int(dut.TermCount.value)

        self._weights = [0] * self._TermCount
        self._window  = [0] * self._TermCount

    def consume(self):
        packed_windows = int(self._window_i.value.integer)
        packed_weights = int(self._weights_i.value.integer)

        # Weights are always signed, unpack_terms handles this correctly
        self._weights = unpack_terms(packed_weights, self._WeightBits, self._TermCount)
        
        # Windows: Use the same helper! 
        # If InBits == 1, it returns [0, 1...]. If > 1, it returns signed ints.
        self._window = unpack_terms(packed_windows, self._InBits, self._TermCount)

        # Apply Bipolar mapping ONLY if 1-bit
        if self._InBits == 1:
            window_vals = [1 if x == 1 else -1 for x in self._window]
        else:
            window_vals = self._window

        if self._WeightBits == 1:
            weight_vals = [1 if x == 1 else -1 for x in self._weights]
        else:
            weight_vals = self._weights
            
        addends = [w * x for w, x in zip(weight_vals, window_vals)]
        return sign_extend(sum(addends), self._OutBits)

    def produce(self, expected):
        assert_resolvable(self._sum_o)
        got = sign_extend(int(self._sum_o.value.integer), self._OutBits)
        exp = int(expected)

        if sim_verbose():
            print(f"Expected: {exp}, Got: {got}, window: {self._window}, weights: {self._weights}")
        
        assert got == exp, (
            f"Mismatch. Expected {exp}, got {got}"
        )
    
async def comb_step(dut, model, windows, weights):
    in_bits     = int(dut.InBits.value)
    weight_bits = int(dut.WeightBits.value)
    dut.window_i.value  = pack_terms(windows, in_bits)
    dut.weights_i.value = pack_terms(weights, weight_bits)

    await Timer(Decimal(1), units="step")

    expected = model.consume()
    model.produce(expected)

@cocotb.test
async def single_test(dut):
    '''Test windows and weights of all ones.'''
    term_count = int(dut.TermCount.value)
    model = MacModel(dut)

    windows = [1] * term_count
    weights = [1] * term_count
    await comb_step(dut, model, windows, weights)

@cocotb.test
async def full_bw_test(dut):
    '''Test windows and weights of random signed values.'''
    in_bits     = int(dut.InBits.value)
    weight_bits = int(dut.WeightBits.value)
    term_count  = int(dut.TermCount.value)
    model       = MacModel(dut)

    for _ in range(10):
        windows = gen_input_channels(in_bits, term_count)
        weights = gen_input_channels(weight_bits, term_count)
        await comb_step(dut, model, windows, weights)