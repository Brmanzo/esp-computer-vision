# test_filter.py
import os
from   pathlib import Path
import pytest

from util.gen_inputs import gen_random_signed
from util.bitwise    import sign_extend, pack_channels, unpack_channels
from util.utilities  import runner, lint, assert_resolvable, clock_start_sequence, \
                            sim_verbose, reset_sequence, load_tests_from_csv, auto_unpack
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import FallingEdge
from   cocotb.result import SimTimeoutError
   
import random
random.seed(50)

timescale = "1ps/1ps"
@pytest.fixture
def use_dsp(request):
    return request.config.getoption("--dsp")

def acc_width(in_bits: int, weight_bits: int, kernel_width: int, in_channels: int, bias_bits: int, unsigned: int = 0) -> int:
    kernel_area = kernel_width * kernel_width

    # 1. Calculate the magnitude of the worst-case convolution
    if in_bits == 1:
        max_input_mag = 1      # bipolar {-1, +1}
    elif unsigned:
        max_input_mag = (1 << in_bits) - 1
    else:
        max_input_mag = 1 << (in_bits - 1)

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

auto_rules = [
    ("AccBits", "AccBits", lambda InBits, WeightBits, KernelWidth, InChannels, BiasBits, Unsigned: acc_width(InBits, WeightBits, KernelWidth, InChannels, BiasBits, Unsigned)),
    ("OutBits", "OutBits", lambda InBits, WeightBits, KernelWidth, InChannels, BiasBits, Unsigned: acc_width(InBits, WeightBits, KernelWidth, InChannels, BiasBits, Unsigned))
]

gen_rules = [
    ("ShiftBits", lambda AccBits, OutBits: max(0, int(AccBits) - int(OutBits)))
]

TEST_CASES = load_tests_from_csv(os.path.join(tbpath, "test_cases.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)

def test_each(test_name, simulator, use_dsp,
              InBits, WeightBits, BiasBits, KernelWidth, InChannels, AccBits, OutBits, Unsigned, ShiftBits):

    if simulator == "icarus" and use_dsp:
        pytest.skip("Icarus Verilog has issues with string parameters for ROM initialization")

    # Skip problematic parallel 1-bit multi-channel configs in Icarus
    if simulator == "icarus" and not use_dsp and InBits == 1 and InChannels > 1:
        pytest.skip("Icarus Verilog misinterprets 3D array ports for parallel multi-channel 1-bit activations")

    # Skip if DSP implementation cannot handle the bit-width
    if simulator == "icarus" and int(AccBits) > 20:
        pytest.skip(f"Icarus Verilog is unreliable with large accumulation widths ({AccBits} bits)")

    if use_dsp:
        # filter_dsp uses a 32-bit accumulator (SB_MAC16)
        # Required bits: max(OutBits, WeightBits + InBits + log2(InChannels * KernelArea))
        import math
        req_bits = max(OutBits, WeightBits + InBits + math.ceil(math.log2(InChannels * KernelWidth * KernelWidth)))
        if req_bits > 32:
            pytest.skip(f"filter_dsp accumulator (32 bits) too small for required {req_bits} bits")

    # This line must be first
    parameters = {
        "InBits": InBits,
        "WeightBits": WeightBits,
        "BiasBits": BiasBits,
        "KernelWidth": KernelWidth,
        "InChannels": InChannels,
        "AccBits": AccBits,
        "OutBits": OutBits,
        "Unsigned": Unsigned,
        "ShiftBits": ShiftBits
    }
    
    # Generate random weights/biases for this test run
    # For sequential ROM, these must be fixed for the duration of the simulation
    from util.gen_inputs import gen_random_signed
    from util.bitwise import pack_terms
    
    term_count = int(KernelWidth)**2 * int(InChannels)
    raw_weights = [gen_random_signed(int(WeightBits), random) for _ in range(term_count)]
    Weights = pack_terms(raw_weights, int(WeightBits))
    Biases  = gen_random_signed(int(BiasBits), random)

    # Sequential ROM Implementation: Use hex-based injection
    custom_work_dir = None
    if use_dsp:
        from util.utilities import inject_weights_and_biases, get_param_string
        # Use the same unique path string that the runner will use, appending test_name to prevent caching stale headers
        param_str = get_param_string(parameters) + f"_{test_name}"
        # Add GEN_DSP for the wrapper
        parameters["GEN_DSP"] = 1
        # inject_weights_and_biases returns the work_dir where headers are stored
        custom_work_dir = inject_weights_and_biases(
            simulator=simulator, parameters=parameters, param_str=param_str, 
            tbpath=tbpath, test_class="sequential", Weights=Weights, Biases=Biases, 
            weight_bits=WeightBits, bias_bits=BiasBits, weight_count=term_count, 
            layer=0, dsp_count=1
        )
    else:
        # Parallel version: we still need to inject weights/biases for the wrapper
        from util.utilities import inject_weights_and_biases, get_param_string
        param_str = get_param_string(parameters) + f"_{test_name}"
        parameters["GEN_DSP"] = 0
        custom_work_dir = inject_weights_and_biases(
            simulator=simulator, parameters=parameters, param_str=param_str, 
            tbpath=tbpath, test_class="parallel", Weights=Weights, Biases=Biases, 
            weight_bits=WeightBits, bias_bits=BiasBits, weight_count=term_count, 
            layer=0, dsp_count=0
        )

    filelist = "filter_dsp.json" if use_dsp else "filter.json"
    wrapper_path = os.path.join(tbpath, "tb_filter.sv")
    
    runner(simulator, timescale, tbpath, parameters, 
           testname=test_name, pymodule="test_filter", filelist=filelist,
           toplevel_override="tb_filter", extra_sources=[wrapper_path],
           work_dir=custom_work_dir, includes=[custom_work_dir])

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, BiasBits, KernelWidth, InChannels, AccBits, OutBits, Unsigned, ShiftBits", 
                         [( 1, 2, 8, 3, 1, acc_width( 1, 2, 3,  1, 8, 0), acc_width( 1, 2, 3,  1, 8, 0), 0, 0)])
def test_lint(simulator, use_dsp, InBits, WeightBits, BiasBits, KernelWidth, InChannels, AccBits, OutBits, Unsigned, ShiftBits):
    # This line must be first
    parameters = {
        "InBits": InBits,
        "WeightBits": WeightBits,
        "BiasBits": BiasBits,
        "KernelWidth": KernelWidth,
        "InChannels": InChannels,
        "AccBits": AccBits,
        "OutBits": OutBits,
        "Unsigned": Unsigned,
        "ShiftBits": ShiftBits
    }
    filelist = "filter_dsp.json" if use_dsp else "filter.json"
    lint(simulator, timescale, tbpath, parameters, filelist=filelist)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, BiasBits, KernelWidth, InChannels, AccBits, OutBits, Unsigned, ShiftBits", 
                         [( 1, 2, 8, 3, 1, acc_width( 1, 2, 3,  1, 8, 0), acc_width( 1, 2, 3,  1, 8, 0), 0, 0)])
def test_style(simulator, use_dsp, InBits, WeightBits, BiasBits, KernelWidth, InChannels, AccBits, OutBits, Unsigned, ShiftBits):
    # This line must be first
    parameters = {
        "InBits": InBits,
        "WeightBits": WeightBits,
        "BiasBits": BiasBits,
        "KernelWidth": KernelWidth,
        "InChannels": InChannels,
        "AccBits": AccBits,
        "OutBits": OutBits,
        "Unsigned": Unsigned,
        "ShiftBits": ShiftBits
    }
    filelist = "filter_dsp.json" if use_dsp else "filter.json"
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"], filelist=filelist)

class FilterModel:
    def __init__(self, dut):
        self._dut = dut
        self._InBits = int(dut.InBits.value)
        self._OutBits = int(dut.OutBits.value)
        self._AccBits = int(dut.AccBits.value)  
        self._WeightBits = int(dut.WeightBits.value)
        self._BiasBits = int(dut.BiasBits.value)
        self._InChannels = int(dut.InChannels.value)
        self._KernelArea = int(dut.KernelWidth.value)**2
        unsigned_obj = getattr(dut, "Unsigned", None)
        self._Unsigned = int(unsigned_obj.value) if unsigned_obj is not None else 0
        shift_obj = getattr(dut, "ShiftBits", None)
        self._ShiftBits = int(shift_obj.value) if shift_obj is not None else 0

    def consume(self):
        p_win  = int(self._dut.windows_i.value)
        # Sequential implementation: always recover from environment for consistency with wrapper
        p_wgt  = int(os.environ["INJECTED_WEIGHTS_0_INT"], 0)
        p_bias = sign_extend(int(os.environ["INJECTED_BIASES_0_INT"], 0), self._BiasBits)

        # Use unified unpacking (assuming utilities are in bitwise.py)
        weights = unpack_channels(p_wgt, self._WeightBits, self._InChannels, self._KernelArea)
        
        if self._InBits == 1:
            windows = unpack_channels(p_win, 1, self._InChannels, self._KernelArea, signed=False)
        else:
            windows = unpack_channels(p_win, self._InBits, self._InChannels, self._KernelArea, signed=not self._Unsigned)

        total = 0
        for ch in range(self._InChannels):
            for i in range(self._KernelArea):
                win = windows[ch][i]
                wgt = weights[ch][i]
                
                # Input bipolar mapping
                if self._InBits == 1:
                    win = 1 if win == 1 else -1
                elif self._Unsigned:
                    win = win & ((1 << self._InBits) - 1)
                
                if self._WeightBits == 1:
                    wgt = 1 if wgt == 1 else -1
                total += win * wgt
        # Add bias
        total += p_bias
        # match rtl logic: (sum_d > 0) ? 1 : 0
        if self._OutBits == 1:
            exp = 1 if total > 0 else 0
        elif self._OutBits == 2:
            exp = 1 if total > 0 else (-1 if total < 0 else 0)
        elif self._OutBits >= self._AccBits:
            exp = sign_extend(total, self._OutBits)
        else:
            # MSB Truncation with ReLU and Saturation (matching output_encoder.sv)
            if total < 0:
                exp = 0
            else:
                shift = self._ShiftBits
                truncated = total >> shift
                max_val = (1 << self._OutBits) - 1
                if truncated > max_val:
                    exp = max_val
                else:
                    exp = truncated
        
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
        if sim_verbose():
            print(f"got: {got}, expected: {expected}")
        assert got == expected, f"Mismatch. Expected {expected}, got {got}"
        
class RandomDataGenerator:
    def __init__(self, dut):
        self._dut         = dut
        self._in_bits     = int(dut.InBits.value)
        self._weight_bits = int(dut.WeightBits.value)
        self._InChannels  = int(dut.InChannels.value)
        self._BiasBits    = int(dut.BiasBits.value)
        self._term_count  = int(dut.KernelWidth.value) * int(dut.KernelWidth.value)
        unsigned_obj = getattr(dut, "Unsigned", None)
        self._Unsigned = int(unsigned_obj.value) if unsigned_obj is not None else 0

    def generate(self):
        from util.bitwise import pack_terms
        # 1. Create Raw Data Structure
        if self._Unsigned:
            from util.gen_inputs import gen_random_unsigned
            windows = [[gen_random_unsigned(self._in_bits, random) for _ in range(self._term_count)] 
                       for _ in range(self._InChannels)]
        else:
            windows = [[gen_random_signed(self._in_bits, random) for _ in range(self._term_count)] 
                       for _ in range(self._InChannels)]
        
        # In DSP mode, weights and biases are fixed in ROM/Parameters
        if "INJECTED_WEIGHTS_0_INT" in os.environ:
            p_wgt = int(os.environ["INJECTED_WEIGHTS_0_INT"], 0)
            from util.bitwise import unpack_channels
            weights = unpack_channels(p_wgt, self._weight_bits, self._InChannels, self._term_count)
            if hasattr(self._dut, "Biases"):
                bias = int(self._dut.Biases.value)
            elif hasattr(self._dut, "Bias"):
                bias = int(self._dut.Bias.value)
            else:
                bias = 0
        else:
            weights = [[gen_random_signed(self._weight_bits, random) for _ in range(self._term_count)] 
                       for _ in range(self._InChannels)]
            bias    = gen_random_signed(self._BiasBits, random)

        # 2. Use unified packers (flattened)
        flat_windows_list = [val for ch in windows for val in ch]
        flat_weights_list = [val for ch in weights for val in ch]
        
        packed_windows = pack_terms(flat_windows_list, self._in_bits)
        packed_weights = pack_terms(flat_weights_list, self._weight_bits)

        return [packed_windows, packed_weights, bias], [windows, weights]

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
    
    # Only drive weights_i if it's a port on the DUT (Parallel implementation)
    data_pins = [dut.windows_i]
    if hasattr(dut, "weights_i"):
        data_pins.append(dut.weights_i)
    if hasattr(dut, "bias_i"):
        data_pins.append(dut.bias_i)
        
    im = InputModel(dut, data_gen, RateGenerator(dut, in_rate), test_length, data_pins=data_pins)

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
    timeout_ns = int(((test_length + 1000) / slow) * 10) # Assuming 10ns clock
    
    # Scale timeout for sequential DSP implementation
    if hasattr(dut, "GEN_DSP") and dut.GEN_DSP.value == 1:
        total_terms = int(dut.InChannels.value) * (int(dut.KernelWidth.value) ** 2)
        neurons_per_dsp = int(dut.OutChannels.value)
        timeout_ns *= (total_terms * neurons_per_dsp)
    # 5. Wait for the OutputModel to hit 'test_length'
    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert False, f"Test timed out. Expected {test_length} outputs."

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