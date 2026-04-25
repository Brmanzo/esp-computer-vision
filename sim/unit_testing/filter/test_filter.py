# test_filter.py
from   pathlib import Path
import pytest

from util.gen_inputs import gen_random_signed
from util.bitwise    import sign_extend
from util.utilities  import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence, delay_cycles
from util.components import ModelRunner, RateGenerator
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import RisingEdge, FallingEdge, with_timeout
from   cocotb.result import SimTimeoutError
   
import random
random.seed(50)

timescale = "1ps/1ps"

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
            terms[ch].append(raw)
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

tests = ['reset_test'
        ,'inout_fuzz_test'
        ,'in_fuzz_test'
        ,'out_fuzz_test'
        ,'full_bw_test']

# Test that binary tree can accomodate 
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits, WeightBits, KernelWidth, InChannels, AccBits, OutBits",
    [( 1, 2, 3,  1, acc_width( 1, 2, 3,  1), 1),
     ( 1, 2, 2,  2, acc_width( 1, 2, 2,  2), acc_width( 1, 2, 2,  2)),
     ( 1, 3, 3,  1, acc_width( 1, 3, 3,  1), 1),
     ( 8, 2, 4,  1, acc_width( 8, 2, 4,  1), acc_width( 8, 2, 4,  1)),
     ( 8, 2, 3,  2, acc_width( 8, 2, 3,  2), 1),
     ( 8, 2, 3, 16, acc_width( 8, 2, 3, 16), 1),
     ( 1, 2, 3, 32, acc_width( 1, 2, 3, 32), acc_width( 1, 2, 3, 32)),
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

        self._windows_i = dut.windows_i
        self._weights_i = dut.weights_i

        self._InBits      = int(dut.InBits.value)
        self._OutBits     = int(dut.OutBits.value)
        self._WeightBits  = int(dut.WeightBits.value)
        self._AccBits     = int(dut.AccBits.value)
        self._InChannels  = int(dut.InChannels.value)
        self._KernelWidth = int(dut.KernelWidth.value)
        self._KernelArea  = self._KernelWidth * self._KernelWidth

        self._weights = None
        self._windows = None

    def consume(self):
        packed_windows = int(self._windows_i.value.integer)
        packed_weights = int(self._weights_i.value.integer)

        weights = unpack_inputs(packed_weights, self._WeightBits, self._InChannels, self._KernelArea)
        if self._InBits == 1:
            windows = unpack_unsigned_inputs(packed_windows, self._InBits, self._InChannels, self._KernelArea)
        else:
            windows = unpack_inputs(packed_windows, self._InBits, self._InChannels, self._KernelArea)

        total = 0
        for ch in range(self._InChannels):
            for i in range(self._KernelArea):
                win = windows[ch][i]
                wgt = weights[ch][i]
                # If input is 1-bit, treat as bipolar {-1, +1} instead of unsigned {0, 1}.
                if self._InBits == 1:
                    win = 1 if win == 1 else -1
                total += win * wgt

        # Encode binary output activation if requested
        if self._OutBits == 1:
            exp = 1 if total > 0 else 0
        else:
            exp = trunc_signed(total, self._OutBits)
        
        return (exp, windows, weights)

    def produce(self, expected):
        assert_resolvable(self._data_o)

        expected, windows, weights = expected

        # If output is single-bit, treat as unsigned 0/1 based on sign of total.
        if self._OutBits == 1:
            got = int(self._data_o.value.integer) & 1
        # For multi-bit outputs, we need to sign-extend the value from the DUT before comparing.
        else:
            got = sign_extend(int(self._data_o.value.integer), self._OutBits)

        print(f"Expected: {expected}, Got: {got}, windows: {windows}, weights: {weights}")
        assert got == expected, f"Mismatch. Expected {expected}, got {got}"


class OutputModel():
    def __init__(self, dut, g, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut

        self._generator = g
        self._length = l

        self._coro = None

        self._nout = 0

    def start(self):
        """ Start Output Model """
        if self._coro is not None:
            raise RuntimeError("Output Model already started")
        self._coro = cocotb.start_soon(self._run())

    def stop(self) -> None:
        """ Stop Output Model """
        if self._coro is None:
            raise RuntimeError("Output Model never started")
        self._coro.kill()
        self._coro = None

    async def wait(self, t):
        if self._coro is None:
            raise RuntimeError("Output Model never started")
        await with_timeout(self._coro, t, 'ns')

    def nproduced(self):
        return self._nout

    async def _run(self):
        """ Output Model Coroutine"""

        self._nout = 0
        clk_i = self._clk_i
        ready_i = self._dut.ready_i
        rst_i = self._dut.rst_i
        valid_o = self._dut.valid_o

        await FallingEdge(clk_i)

        if(not (rst_i.value.is_resolvable and rst_i.value == 0)):
            await FallingEdge(rst_i)

        # Precondition: Falling Edge of Clock
        while self._nout < self._length:
            consume = self._generator.generate()
            success = 0
            ready_i.value = consume

            # Wait until valid
            while(consume and not success):
                await RisingEdge(clk_i)
                assert_resolvable(valid_o)
                #assert valid_o.value.is_resolvable, f"Unresolvable value in valid_o (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."

                success = True if (valid_o.value == 1) else False
                if (success):
                    self._nout += 1

            await FallingEdge(clk_i)
        return self._nout

class InputModel():
    def __init__(self, dut, generator, rate, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut   = dut

        self._rate      = rate
        self._generator = generator
        self._length    = l
        self._coro      = None
        self._nin       = 0

    def start(self):
        """ Start Input Model """
        if self._coro is not None:
            raise RuntimeError("Input Model already started")
        self._coro = cocotb.start_soon(self._run())

    def stop(self) -> None:
        """ Stop Input Model """
        if self._coro is None:
            raise RuntimeError("Input Model never started")
        self._coro.kill()
        self._coro = None

    async def wait(self, t):
        if self._coro is None:
            raise RuntimeError("Input Model never started")
        await with_timeout(self._coro, t, 'ns')

    def nconsumed(self):
        return self._nin

    async def _run(self):
        """ Input Model Coroutine"""

        self._nin = 0
        clk_i   = self._clk_i
        rst_i   = self._dut.rst_i
        ready_o = self._dut.ready_o
        valid_i = self._dut.valid_i
        weights_i = self._dut.weights_i
        windows_i = self._dut.windows_i

        in_bits     = int(self._dut.InBits.value)
        weight_bits = int(self._dut.WeightBits.value)
        in_channels = int(self._dut.InChannels.value)
        term_count  = int(self._dut.KernelWidth.value) ** 2

        await delay_cycles(self._dut, 1, False)

        if(not (rst_i.value.is_resolvable and rst_i.value == 0)):
            await FallingEdge(rst_i)

        await delay_cycles(self._dut, 2, False)

        # generate data
        windows, weights = self._generator.generate()

        # Precondition: Falling Edge of Clock
        while self._nin < self._length:
            produce = bool(self._rate.generate())
            valid_i.value = int(produce)

            # 1. Pack the lists of windows and weights into the appropriate integer format for the DUT
            windows_i.value = pack_inputs(windows, in_bits, in_channels, term_count)
            weights_i.value = pack_inputs(weights, weight_bits, in_channels, term_count)

            # Wait for handshake if producing, otherwise just advance a cycle
            success = False

            # Wait until ready
            while(produce and not success):
                await RisingEdge(clk_i)
                assert_resolvable(ready_o)

                success = True if (ready_o.value == 1) else False
                if (success):
                    self._nin += 1
                    
                    # 3. Generate the NEXT data sample only after a successful transfer!
                    windows, weights = self._generator.generate()

            await FallingEdge(clk_i)
            
        valid_i.value = 0
        return self._nin
        
class RandomDataGenerator:
    def __init__(self, dut):
        self._in_bits     = int(dut.InBits.value)
        self._weight_bits = int(dut.WeightBits.value)
        self._InChannels  = int(dut.InChannels.value)
        self._term_count  = int(dut.KernelWidth.value) * int(dut.KernelWidth.value)

    def generate(self):

        windows = [[gen_random_signed(self._in_bits, random)     for _ in range(self._term_count)] for _ in range(self._InChannels)]
        weights = [[gen_random_signed(self._weight_bits, random) for _ in range(self._term_count)] for _ in range(self._InChannels)]

        return windows, weights
    
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
    im = InputModel(dut, data_gen, RateGenerator(dut, in_rate), test_length)

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