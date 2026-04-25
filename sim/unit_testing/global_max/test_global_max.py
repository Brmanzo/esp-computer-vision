# test_global_max.py
from   pathlib import Path
import pytest

from util.utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence, delay_cycles
from util.bitwise   import pack_terms, unpack_terms
from util.components import ModelRunner, RateGenerator, InputModel
from util.gen_inputs import gen_input_channels
tbpath = Path(__file__).parent

import cocotb
from   cocotb.utils import get_sim_time
from   cocotb.triggers import RisingEdge, FallingEdge, with_timeout
from   cocotb.result import SimTimeoutError
   
import random
random.seed(42)

timescale = "1ps/1ps"

tests = ['reset_test'
        ,'single_test'
        ,'inout_fuzz_test'
        ,'in_fuzz_test'
        ,'out_fuzz_test'
        ,'full_bw_test']

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits, TermCount, InChannels", [
    (1, 16, 10),
    (8, 9, 1),
    (16, 17, 2),
    (32, 33, 4)
])
def test_each(test_name, simulator, InBits, TermCount, InChannels):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_global_max")

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
def test_all(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, pymodule="test_global_max")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, pymodule="test_global_max")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"], pymodule="test_global_max")

class GlobalMaxModel():
    def __init__(self, dut):
        self._dut = dut
        self._data_i = dut.data_i
        self._data_o = dut.data_o

        self._in_bits     = int(dut.InBits.value)
        self._out_bits    = int(dut.OutBits.value)
        self._in_channels = int(dut.InChannels.value)
        self._terms       = int(dut.TermCount.value)

        self._term_counter = 0
        self._current_max  = None

    
    def consume(self):
        assert_resolvable(self._data_i)

        packed_in = self._data_i.value.integer
        x = unpack_terms(packed_in, self._in_bits, self._in_channels)

        if self._term_counter == 0:
            self._current_max = x[:]
        else:
            assert self._current_max is not None
            self._current_max = [
                max(self._current_max[ch], x[ch])
                for ch in range(self._in_channels)
            ]

        self._term_counter += 1

        if self._term_counter == self._terms:
            expected_max = self._current_max[:]

            self._term_counter = 0
            self._current_max = None
            
            return tuple(expected_max)

    def produce(self, expected):
        assert_resolvable(self._data_o)

        packed_out = self._data_o.value.integer
        got = unpack_terms(packed_out, self._out_bits, self._in_channels)

        expected_list = list(expected)

        print(f"Produced output {got}, expected {expected_list} at time {get_sim_time(units='ns')}ns")
        assert got == expected_list, (
            f"Output mismatch. Expected {expected_list}, got {got}"
        )

class RandomDataGenerator:
    def __init__(self, dut):
        self._dut = dut
        self._in_bits = int(self._dut.InBits.value)
        self._in_channels = int(self._dut.InChannels.value)

    def generate(self):
        # 1. Generate the raw list of integers
        raw_list = gen_input_channels(self._in_bits, self._in_channels)
        
        # 2. Pack them into a single big integer for the DUT
        packed_val = pack_terms(raw_list, self._in_bits)
        
        # 3. Return (Value_for_Pins, Value_for_Model_Callback)
        return packed_val, raw_list

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

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

@cocotb.test
async def single_test(dut):
    T = int(dut.TermCount.value)

    model = GlobalMaxModel(dut)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, 1), 1)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, 1), T)

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    timeout_ns = 200
    await om.wait(timeout_ns)

async def rate_tests(dut, in_rate, out_rate):
    """Input random data elements at 100% line rate"""

    eg = RandomDataGenerator(dut)
    T = int(dut.TermCount)
    l_in = 10*T
    l_out = l_in // T

    m = ModelRunner(dut, GlobalMaxModel(dut))
    om = OutputModel(dut, RateGenerator(dut, out_rate), l_out)
    im = InputModel(dut, eg, RateGenerator(dut, in_rate), l_in)

    clk_i = dut.clk_i
    rst_i = dut.rst_i

    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    await RisingEdge(dut.ready_i)
    await RisingEdge(dut.clk_i)

    slow = min(in_rate, out_rate)
    slow = max(slow, 0.05)
    timeout_ns        = int((l_in + 500) / slow)

    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, (
            f"Test timed out. Expected {l_out} outputs from {l_in} inputs "
            f"with TermCount={T}, in_rate={in_rate}, out_rate={out_rate}"
    )

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
