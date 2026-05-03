# test_packer.py
import os
from   pathlib import Path
import pytest

from util.utilities import runner, lint, assert_resolvable, clock_start_sequence, \
                           sim_verbose, reset_sequence, load_tests_from_csv, auto_unpack
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
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
        ,'flush_test']

TEST_CASES = load_tests_from_csv(os.path.join(tbpath, "test_cases.csv"))
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)
def test_each(test_name, simulator,
              UnpackedWidth, PackedNum):
    # This line must be first
    parameters = dict(locals())
    parameters.pop('test_name', None)
    parameters.pop('simulator', None)
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_packer")

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)
def test_all(simulator, UnpackedWidth, PackedNum):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, pymodule="test_packer")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, pymodule="test_packer")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"], pymodule="test_packer")

class PackerModel():
    def __init__(self, dut):
        self._dut = dut

        self._unpacked_i = dut.unpacked_i
        self._flush_i    = dut.flush_i

        self._PackedNum = int(dut.PackedNum.value)
        self._PackedWidth = int(dut.PackedWidth.value)
        self._UnpackedWidth = int(dut.UnpackedWidth.value)
        
        self._step = 0
        self._acc = 0
        self._enqs = 0
        self._deqs = 0

    def consume(self):
        """Called by ModelRunner when it detects an input handshake."""
        # 1. Read data from pins directly
        # Use the handles we saved in __init__
        assert_resolvable(self._unpacked_i)
        assert_resolvable(self._flush_i)
        
        u = int(self._unpacked_i.value) & ((1 << self._UnpackedWidth) - 1)
        flush = int(self._flush_i.value) & 1
        
        # 2. Packing logic
        self._acc |= (u << (self._UnpackedWidth * self._step))
        self._enqs += 1

        completed = (self._step == self._PackedNum - 1) or bool(flush)
        
        if completed:
            expected = (self._acc & ((1 << self._PackedWidth) - 1))
            self._acc = 0
            self._step = 0
            return expected  # Return value goes into ModelRunner's queue
        else:
            self._step += 1
            return None      # No output expected yet

    def produce(self, expected):
        assert_resolvable(self._dut.packed_o)
        got = int(self._dut.packed_o.value) & ((1 << self._PackedWidth) - 1)
        self._deqs += 1
        
        if sim_verbose():
            print (f"Produced output #{self._deqs}: 0x{got:X}, expected 0x{expected:X}")
        assert got == expected, f"Mismatch out #{self._deqs}: exp 0x{expected:X}, got 0x{got:X}"

class RandomDataGenerator():
    def __init__(self, dut, flush_rate):
        self._width = int(dut.UnpackedWidth.value)
        self._flush_rate = flush_rate
        self._mask = (1 << self._width) - 1

    def generate(self):
        data = gen_random_unsigned(self._width, random) & self._mask
        
        # Determine flush bit
        flush = 1 if (self._flush_rate > 0 and random.random() < self._flush_rate) else 0

        return [data, flush], [data, flush]

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

    eg = RandomDataGenerator(dut, flush_rate=0)
    l_out = 1
    l_in = l_out * dut.PackedNum.value
    rate = 1

    timeout = max(l_out, l_in) * int(1/rate) * int(1/rate) 

    m = ModelRunner(dut, PackerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), l_in, 
                    data_pins=[dut.unpacked_i, dut.flush_i])

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i
    flush_i = dut.flush_i

    ready_i.value = 0
    valid_i.value = 0
    flush_i.value = 0

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

    timeout_cycles = int((l_in + l_out) * (1/rate) * dut.PackedNum.value) + 50
    timeout = False
    try:
        await om.wait(timeout_cycles)
    except:
        timeout = True
    assert not timeout, "Error! Maximum latency expected for this circuit is one cycle."

    dut.valid_i.value = 0
    dut.ready_i.value = 0

async def rate_tests(dut, in_rate, out_rate, flush_rate=0):
    # 1. Setup Data and Model
    eg = RandomDataGenerator(dut, flush_rate)
    l_out = 50
    l_in = l_out * int(dut.PackedNum.value)

    # 2. Instantiate Generic ModelRunner
    # It will find valid_i/ready_o and valid_o/ready_i automatically
    m = ModelRunner(dut, PackerModel(dut))

    # 3. Instantiate Custom OutputModel (if you prefer your local one) 
    # OR use a generic InputModel to drive ready_i
    om = OutputModel(dut, RateGenerator(dut, out_rate), l_out)

    # 4. Instantiate Generic InputModel
    im = InputModel(
        dut, eg, RateGenerator(dut, in_rate), l_in, 
        data_pins=[dut.unpacked_i, dut.flush_i] # Overrides the default 'data_i'
    )

    # Standard Boot Sequence
    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait for the handshake driver to finish
    slow = min(in_rate, out_rate, 0.05)
    timeout_ns = int((l_in * 10 + 500) / slow)
    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, f"Timed out. Got {om.nproduced()} / {l_out} outputs."

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

@cocotb.test
async def flush_test(dut):
    """Input random data elements at 100% line rate"""

    eg = RandomDataGenerator(dut, flush_rate=0.1)
    l_out = 50
    l_in = l_out * dut.PackedNum.value
    rate = 1

    timeout = max(l_out, l_in) * int(1/rate) * int(1/rate) 

    m = ModelRunner(dut, PackerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), l_in, 
                    data_pins=[dut.unpacked_i, dut.flush_i])

    clk_i = dut.clk_i
    rst_i = dut.rst_i

    ready_i = dut.ready_i
    valid_i = dut.valid_i
    flush_i = dut.flush_i

    ready_i.value = 0
    valid_i.value = 0
    flush_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    await RisingEdge(dut.ready_i)
    await RisingEdge(dut.clk_i)

    CLK_NS = 10
    timeout_cycles = int((l_in + l_out) * (1/rate) * dut.PackedNum.value) + 50
    timeout_ns = timeout_cycles * CLK_NS
    await om.wait(timeout_ns)