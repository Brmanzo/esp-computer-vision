# test_mac.py
from   decimal import Decimal
from   pathlib import Path
import pytest

from util.utilities import runner, lint, assert_resolvable, sign_extend
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import Timer
   
import random
random.seed(50)

timescale = "1ps/1ps"

tests = ['single_test'
        ,'full_bw_test']

def pack_inputs(input, in_bits):
    packed = 0
    mask = (1 << in_bits) - 1
    for i, x in enumerate(input):
        packed |= (x & mask) << (i * in_bits)
    return packed

def unpack_inputs(packed, in_bits, term_count):
    terms = []
    mask = (1 << in_bits) - 1
    for i in range(term_count):
        raw = (packed >> (i * in_bits)) & mask
        terms.append(sign_extend(raw, in_bits))
    return terms

def unpack_unsigned_inputs(packed, in_bits, term_count):
    terms = []
    mask = (1 << in_bits) - 1
    for i in range(term_count):
        raw = (packed >> (i * in_bits)) & mask
        terms.append(raw) # NO sign extension!
    return terms

def trunc_signed(value: int, width: int) -> int:
    return sign_extend(value, width)

def output_width(in_width: int, weight_width: int, term_count: int) -> str:
    '''Calculates proper output width for mixed signed/unsigned MAC.'''
    max_in = 1 if in_width == 1 else (1 << in_width) - 1 
    max_weight = 1 << (weight_width - 1) 
    max_sum = term_count * max_in * max_weight
    abs_bits = max_sum.bit_length()
    return str(abs_bits + 1)

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits, WeightBits, TermCount, OutBits",
    [( 1,  2,  2, output_width( 1,  2,  2)),
     ( 1,  2,  3, output_width( 1,  2,  3)),
     ( 1,  3, 32, output_width( 1,  3, 32)),
     ( 8,  2,  5, output_width( 8,  2,  5)),
     ( 8,  2, 16, output_width( 8,  2, 16)),
     ( 8,  2, 32, output_width( 8,  2, 32)),
    ],
)

def test_each(test_name, simulator, InBits, WeightBits, TermCount, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_mac")

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, TermCount, OutBits", 
                         [(1, 2, 2, output_width(1, 2, 2)), (2, 2, 3, output_width(2, 2, 3))])
def test_lint(simulator, InBits, WeightBits, TermCount, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, TermCount, OutBits", 
                         [(1, 2, 2, output_width(1, 2, 2)), (2, 2, 3, output_width(2, 2, 3))])
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

    def consume(self): # Remove arguments here!
        # Read the packed integers directly from the DUT pins
        packed_windows = int(self._window_i.value.integer)
        packed_weights = int(self._weights_i.value.integer)

        # Weights are always signed
        self._weights = unpack_inputs(packed_weights, self._WeightBits, self._TermCount)
        
        # Apply {-1,1} encoding if single bit input
        if self._InBits == 1:
            # Unpack as unsigned (0 or 1) so the bipolar list comprehension works
            self._window = unpack_unsigned_inputs(packed_windows, self._InBits, self._TermCount)
            window_vals = [1 if x == 1 else -1 for x in self._window]
        else:
            # Multi-bit windows are SIGNED in hardware, must be signed in Python!
            self._window = unpack_inputs(packed_windows, self._InBits, self._TermCount)
            window_vals = self._window
            
        addends = [w * x for w, x in zip(self._weights, window_vals)]
        return trunc_signed(sum(addends), self._OutBits)

    def produce(self, expected):
        assert_resolvable(self._sum_o)
        got = sign_extend(int(self._sum_o.value.integer), self._OutBits)
        exp = int(expected)

        print(f"Expected: {exp}, Got: {got}, window: {self._window}, weights: {self._weights}")
        assert got == exp, (
            f"Mismatch. Expected {exp}, got {got}"
        )
    
async def comb_step(dut, model, windows, weights):
    in_bits     = int(dut.InBits.value)
    weight_bits = int(dut.WeightBits.value)
    dut.window_i.value  = pack_inputs(windows, in_bits)
    dut.weights_i.value = pack_inputs(weights, weight_bits)

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

    # If 1 bit inputs, generate just zeros and ones
    if in_bits == 1:
        window_lo, window_hi = 0, 1
    # If multi-bit inputs, generate full range of signed values
    else:
        window_lo = -(1 << (in_bits - 1))
        window_hi =  (1 << (in_bits - 1)) - 1

    # Weights are signed two's complement (Min to Max)
    weight_hi =  (1 << (weight_bits - 1)) - 1
    weight_lo = -(1 << (weight_bits - 1))

    for _ in range(10):
        windows = [random.randint(window_lo, window_hi) for _ in range(term_count)]
        weights = [random.randint(weight_lo, weight_hi) for _ in range(term_count)]
        await comb_step(dut, model, windows, weights)