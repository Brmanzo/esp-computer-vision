# test_filter.py
from   pathlib import Path
import pytest

from util.gen_inputs import gen_random_signed
from util.bitwise    import sign_extend, pack_channels, unpack_channels
from util.utilities  import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import FallingEdge
from   cocotb.result import SimTimeoutError
   
import random
random.seed(50)

timescale = "1ps/1ps"

def acc_width(in_bits: int, weight_bits: int, kernel_width: int, in_channels: int, bias_bits: int) -> int:
    kernel_area = kernel_width * kernel_width

    # 1. Calculate the magnitude of the worst-case convolution
    if in_bits == 1:
        max_input_mag = 1      # bipolar {-1, +1}
    else:
        max_input_mag = (1 << in_bits) - 1

    max_weight_mag = 1 << (weight_bits - 1) 
    max_sum_mag = kernel_area * in_channels * max_input_mag * max_weight_mag
    
    # 2. Convert that magnitude to a bit-count (including sign bit)
    conv_bits = max_sum_mag.bit_length() + 1
    
    # 3. The container must be at least as large as conv_bits or bias_bits
    # Then add +1 to allow for the final addition of the two
    acc_width = max(conv_bits, bias_bits) + 1

    return acc_width

tests = ['reset_test'
        ,'inout_fuzz_test'
        ,'in_fuzz_test'
        ,'out_fuzz_test'
        ,'full_bw_test']

# Test that binary tree can accomodate 
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits, WeightBits, BiasBits,KernelWidth, InChannels, AccBits, OutBits",
    [( 1, 2, 8, 3,  1, acc_width( 1, 2, 3,  1, 8), 1),
     ( 1, 2, 8, 2,  2, acc_width( 1, 2, 2,  2, 8), acc_width( 1, 2, 2,  2, 8)),
     ( 1, 3, 8, 3,  1, acc_width( 1, 3, 3,  1, 8), 1),
     ( 8, 2, 8, 4,  1, acc_width( 8, 2, 4,  1, 8), acc_width( 8, 2, 4,  1, 8)),
     ( 8, 2, 8, 3,  2, acc_width( 8, 2, 3,  2, 8), 1),
     ( 8, 2, 8, 3, 16, acc_width( 8, 2, 3, 16, 8), 1),
     ( 1, 2, 8, 3, 32, acc_width( 1, 2, 3, 32, 8), acc_width( 1, 2, 3, 32, 8)),
    ],
)

def test_each(test_name, simulator, InBits, WeightBits, BiasBits, KernelWidth, InChannels, AccBits, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_filter")

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, BiasBits, KernelWidth, InChannels, AccBits, OutBits", 
                         [( 1, 2, 8, 3, 1, acc_width( 1, 2, 3,  1, 8), acc_width( 1, 2, 3,  1, 8))])
def test_lint(simulator, InBits, WeightBits, BiasBits, KernelWidth, InChannels, AccBits, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, BiasBits, KernelWidth, InChannels, AccBits, OutBits", 
                         [( 1, 2, 8, 3, 1, acc_width( 1, 2, 3,  1, 8), acc_width( 1, 2, 3,  1, 8))])
def test_style(simulator, InBits, WeightBits, BiasBits, KernelWidth, InChannels, AccBits, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

class FilterModel:
    def __init__(self, dut):
        self._dut = dut
        self._InBits = int(dut.InBits.value)
        self._OutBits = int(dut.OutBits.value)
        self._WeightBits = int(dut.WeightBits.value)
        self._BiasBits = int(dut.BiasBits.value)
        self._InChannels = int(dut.InChannels.value)
        self._KernelArea = int(dut.KernelWidth.value)**2

    def consume(self):
        p_win  = int(self._dut.windows_i.value)
        p_wgt  = int(self._dut.weights_i.value)
        p_bias = sign_extend(int(self._dut.bias_i.value), self._BiasBits)

        # Use unified unpacking (assuming utilities are in bitwise.py)
        weights = unpack_channels(p_wgt, self._WeightBits, self._InChannels, self._KernelArea)
        
        if self._InBits == 1:
            windows = unpack_channels(p_win, 1, self._InChannels, self._KernelArea, signed=False)
        else:
            windows = unpack_channels(p_win, self._InBits, self._InChannels, self._KernelArea, signed=True)

        total = 0
        for ch in range(self._InChannels):
            for i in range(self._KernelArea):
                win = windows[ch][i]
                wgt = weights[ch][i]
                
                # Input bipolar mapping
                if self._InBits == 1:
                    win = 1 if win == 1 else -1
                total += win * wgt
        print(f"DEBUG: Total before bias: {total}, Bias: {p_bias}")
        # Add bias
        total += p_bias
        # match rtl logic: (sum_d > 0) ? 1 : 0
        if self._OutBits == 1:
            exp = 1 if total > 0 else 0
        else:
            exp = sign_extend(total, self._OutBits)
        
        return [(exp, windows, weights)]

    def produce(self, expected_tuple):
        assert_resolvable(self._dut.data_o)
        expected, windows, weights = expected_tuple
        raw_out = int(self._dut.data_o.value)
  
        # Don't sign_extend if OutBits is 1. 
        # Treat it as a raw logical bit to match the RTL's (sum_d > 0) logic.
        if self._OutBits == 1:
            got = raw_out & 1 
        else:
            got = sign_extend(raw_out, self._OutBits)

        if got != expected:
            print(f"DEBUG: Total Sum was likely {'positive' if expected == 1 else 'zero/negative'}")
            print(f"DEBUG: Raw RTL Bits: {bin(raw_out)}, Expected: {expected}, Got: {got}")
        print(f"got: {got}, expected: {expected}")
        assert got == expected, f"Mismatch. Expected {expected}, got {got}"
        
class RandomDataGenerator:
    def __init__(self, dut):
        self._in_bits     = int(dut.InBits.value)
        self._weight_bits = int(dut.WeightBits.value)
        self._InChannels  = int(dut.InChannels.value)
        self._BiasBits    = int(dut.BiasBits.value)
        self._term_count  = int(dut.KernelWidth.value) * int(dut.KernelWidth.value)

    def generate(self):
        # 1. Create Raw Data Structure
        windows = [[gen_random_signed(self._in_bits, random) for _ in range(self._term_count)] 
                   for _ in range(self._InChannels)]
        weights = [[gen_random_signed(self._weight_bits, random) for _ in range(self._term_count)] 
                   for _ in range(self._InChannels)]
        bias    = gen_random_signed(self._BiasBits, random)

        # 2. Use unified packers
        # Generator returns ([packed_val1, packed_val2], [raw_val1, raw_val2])
        return [pack_channels(windows, self._in_bits), 
                pack_channels(weights, self._weight_bits), 
                bias], [windows, weights]

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)
    
async def rate_tests(dut, in_rate, out_rate, test_length=100):
    """Unified sequential test runner"""
    
    # 1. Initialize the Models
    model = FilterModel(dut)
    m = ModelRunner(dut, model)
    
    data_gen = RandomDataGenerator(dut)
    
    om = OutputModel(dut, RateGenerator(dut, out_rate), test_length)
    im = InputModel(dut, data_gen, RateGenerator(dut, in_rate), test_length, data_pins=[dut.windows_i, dut.weights_i, dut.bias_i])

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    dut.ready_i.value = 0
    dut.valid_i.value = 0

    # 2. Start Clocks and Reset
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)
    await FallingEdge(clk_i)

    # 3. Start the Coroutines
    m.start()
    om.start()
    im.start()

    # 4. Calculate timeout based on rates
    slow = min(in_rate, out_rate)
    slow = max(slow, 0.05) 
    timeout_ns = int(((test_length + 500) / slow) * 10) # Assuming 10ns clock

    # 5. Wait for the OutputModel to hit 'test_length'
    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, f"Test timed out. Expected {test_length} outputs."

@cocotb.test
async def out_fuzz_test(dut):
    await rate_tests(dut, in_rate=1.0, out_rate=0.5)

@cocotb.test
async def in_fuzz_test(dut):
    await rate_tests(dut, in_rate=0.5, out_rate=1.0)

@cocotb.test
async def inout_fuzz_test(dut):
    await rate_tests(dut, in_rate=0.5, out_rate=0.5)

@cocotb.test
async def full_bw_test(dut):
    await rate_tests(dut, in_rate=1.0, out_rate=1.0)