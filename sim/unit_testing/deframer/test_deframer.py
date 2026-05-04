# test_deframer.py
import os
from   pathlib import Path
import pytest

from util.utilities import runner, lint, clock_start_sequence, \
                           reset_sequence, load_tests_from_csv, auto_unpack
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from functional_models.deframer import DeframerModel
from util.gen_inputs import gen_random_unsigned
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import RisingEdge, FallingEdge
from   cocotb.result import SimTimeoutError

import random
random.seed(42)

timescale = "1ps/1ps"

tests = ['reset_test'
        ,'single_test'
        ,'inout_fuzz_test'
        ,'in_fuzz_test'
        ,'out_fuzz_test'
        ,'full_bw_test'
        ,'inout_fuzz_repeat_test'
        ,'in_fuzz_repeat_test'
        ,'out_fuzz_repeat_test'
        ,'full_bw_repeat_test']

TEST_CASES = load_tests_from_csv(os.path.join(tbpath, "test_cases.csv"))
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)
def test_each(test_name, simulator,
              UnpackedWidth, PackedNum, PacketLenElems):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_deframer")

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)
def test_all(simulator,
             UnpackedWidth, PackedNum, PacketLenElems):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, pymodule="test_deframer")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, pymodule="test_deframer")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"], pymodule="test_deframer")

class RandomHeaderGenerator():
    '''After a predefined delay, outputs the deframer bytes, then random data'''
    def __init__(self, dut, initial_delay, repetitions, period):
        self._dut = dut
        self._width_p = dut.PackedWidth.value
        self._initial_delay = initial_delay
        self._header_delay = initial_delay # Cycles until header appears
        self._repetitions = repetitions
        self._period = period
        self._period_remaining = period
        
    def generate(self):
        mask = (1 << self._width_p) - 1
        val = 0

        # Countdown logic
        if self._header_delay > 0:
            self._header_delay -= 1
            val = random.randint(0, mask)
        elif self._header_delay == 0:
            self._header_delay = -1
            val = int(self._dut.HeaderByte0.value) & mask
        elif self._header_delay == -1:
            self._header_delay = -2
            self._period_remaining = self._period
            val = int(self._dut.HeaderByte1.value) & mask
        else:
            # Payload logic
            val = random.randint(0, mask)
            if self._period_remaining > 0:
                self._period_remaining -= 1
            if self._period_remaining == 0:
                self._repetitions -= 1
                if self._repetitions > 0:
                    self._header_delay = self._initial_delay

        return val, val
    
class RandomDataGenerator():
    def __init__(self, dut):
        self._dut = dut
        self._width_p = dut.PackedWidth.value

    def generate(self):
        din = gen_random_unsigned(self._width_p, random)
        return din, din
    
@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

@cocotb.test
async def single_test(dut):
    """Test to transmit a single element in at most two cycles."""

    delay = 10
    header_cycles = 2
    l = 1
    eg = RandomHeaderGenerator(dut, delay, repetitions=1, period=l)
    
    n_in = delay + header_cycles + l
    n_out = l * int(dut.PackedNum.value)
    rate = 1

    timeout = 20000
   
    model = DeframerModel(dut)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, 1), n_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), n_in)

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    # Wait one cycle for reset to start
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)

    im.start()
    await RisingEdge(dut.valid_i)
    await RisingEdge(dut.clk_i)

    timed_out = False
    try:
        await om.wait(timeout + 100)
    except:
        timed_out = True
    assert not timed_out, "Error! Maximum latency expected for this fifo is two cycles."

    dut.valid_i.value = 0
    dut.ready_i.value = 0


async def rate_tests(dut, in_rate, out_rate):
    """Input random data elements at 100% line rate"""
    delay = 10
    header_cycles = 2
    l = 10
    eg = RandomHeaderGenerator(dut, delay, repetitions=1, period=l)
    n_in = delay + header_cycles + l
    n_out = l * int(dut.PackedNum.value)

    m = ModelRunner(dut, DeframerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, out_rate), n_out)
    im = InputModel(dut, eg, RateGenerator(dut, in_rate), n_in)

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
    timeout_ns        = int((n_in + 500) / slow)

    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout_ns} ns, with output rate {out_rate}"

async def repeat_rate_tests(dut, in_rate, out_rate):
    """Input random data elements at 100% line rate"""
    delay = 10
    header_cycles = 2
    repetitions = 3
    packet_len = 10

    eg = RandomHeaderGenerator(dut, delay, repetitions, period=packet_len)
    l = packet_len*repetitions
    n_in  = repetitions * (delay + header_cycles + packet_len)
    n_out = l * int(dut.PackedNum.value)

    m = ModelRunner(dut, DeframerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, out_rate), n_out)
    im = InputModel(dut, eg, RateGenerator(dut, in_rate), n_in)

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
    timeout_ns        = int((n_in + 500) / slow)

    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout_ns} ns, with output rate {out_rate}"

# Standard rate tests
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

# Repeat transmission tests to check for state retention issues across multiple packets
@cocotb.test
async def out_fuzz_repeat_test(dut):
    await repeat_rate_tests(dut, in_rate=1.0, out_rate=0.5)

@cocotb.test
async def in_fuzz_repeat_test(dut):
    await repeat_rate_tests(dut, in_rate=0.5, out_rate=1.0)

@cocotb.test
async def inout_fuzz_repeat_test(dut):
    await repeat_rate_tests(dut, in_rate=0.5, out_rate=0.5)

@cocotb.test
async def full_bw_repeat_test(dut):
    await repeat_rate_tests(dut, in_rate=1.0, out_rate=1.0)