from   pathlib import Path
import pytest

from util.utilities  import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from util.gen_inputs import gen_random_unsigned
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import RisingEdge, FallingEdge, with_timeout
from   cocotb.result import SimTimeoutError
   
import random
random.seed(42)

timescale = "1ps/1ps"
tests = ['reset_test'
         ,'single_test'
         ,'datapath_reset_test'
         ,'datapath_gate_test'
         ,'fill_test'
         ,'fill_empty_test'
         ,'in_fuzz_test'
         ,'out_fuzz_test'
         ,'inout_fuzz_test'
         ,'full_bw_test'
         ]

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits",        [7, 32])
@pytest.mark.parametrize("DatapathReset", [0, 1])
@pytest.mark.parametrize("DatapathGate",  [0, 1])

def test_each(test_name, simulator, InBits, DatapathReset, DatapathGate):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name)

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits",        [7, 32])
@pytest.mark.parametrize("DatapathReset", [0, 1])
@pytest.mark.parametrize("DatapathGate",  [0, 1])

def test_all(simulator, InBits, DatapathReset, DatapathGate):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters)
@pytest.mark.parametrize("simulator",     ["verilator"])

@pytest.mark.parametrize("InBits, DatapathReset, DatapathGate", [(7, 0, 0)])

def test_lint(simulator, InBits, DatapathReset, DatapathGate):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, DatapathReset, DatapathGate", [(7, 0, 0)])

def test_style(simulator, InBits, DatapathReset, DatapathGate):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

async def wait_for_signal(dut, signal, value):
    signal = getattr(dut, signal)
    while(signal.value.is_resolvable and signal.value != value):
        await FallingEdge(dut.clk_i)

async def delay_cycles(dut, ncyc, polarity):
    for _ in range(ncyc):
        if(polarity):
            await RisingEdge(dut.clk_i)
        else:
            await FallingEdge(dut.clk_i)

class ElasticModel():
    def __init__(self, dut):
        self._dut = dut
        self._deqs = 0
        self._enqs = 0

    def consume(self):
        """Called by ModelRunner on a valid input handshake."""
        assert_resolvable(self._dut.data_i)
        
        # Read the bit-accurate value from the bus
        val = int(self._dut.data_i.value)
        self._enqs += 1
        
        # Return the value so ModelRunner can queue it
        return val

    def produce(self, expected):
        """Called by ModelRunner on a valid output handshake."""
        assert_resolvable(self._dut.data_o)
        
        got = int(self._dut.data_o.value)
        
        # The ModelRunner passed us the 'expected' value it popped from its queue
        assert got == expected, f"Error! Value on deque iteration {self._deqs} does not match expected. Expected: {expected}. Got: {got}"
        self._deqs += 1

class SingletonGenerator():
    def __init__(self, dut, v):
        self._value = v

    def generate(self):
        # Return tuple: (packed_vals, raw_vals)
        return self._value, self._value
    
class RandomDataGenerator():
    def __init__(self, dut):
        self._dut = dut

    def generate(self):
        val = gen_random_unsigned(self._dut.InBits.value, random)
        return val, val

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

@cocotb.test
async def datapath_reset_test(dut):
    """Test for DatapathReset"""

    depth_p = 1
    l = depth_p
    rate = 1
    value = (1 << dut.InBits.value) - 1

    m = ModelRunner(dut, ElasticModel(dut))
    om = OutputModel(dut, RateGenerator(dut, 0), l)
    im = InputModel(dut, SingletonGenerator(dut, value), RateGenerator(dut, rate), l)

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
    im.start()

    await RisingEdge(dut.valid_i)
    await RisingEdge(dut.clk_i)

    timeout = False
    try:
        await with_timeout(wait_for_signal(dut, "valid_o", 1), 3, 'ns')
    except:
        timeout = True
    assert not timeout, "Error! Data transfer was not moved to output within expected timeframe."

    await RisingEdge(dut.clk_i)
    assert dut.data_o == value,  f"Error! Value on data_o does not match expected. Expected: {value}. Got: {dut.data_o.value}"

    await FallingEdge(dut.clk_i)
    dut.rst_i.value = 1

    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    if(dut.DatapathReset.value == 1):
        assert dut.data_o == 0,  f"Error! Value on data_o should be zero after reset when DatapathReset == 1. Expected: 0. Got: {dut.data_o.value}"
    else:
        assert dut.data_o == value,  f"Error! Value on data_o should not be zero after reset when DatapathReset == 0. Expected: {value}. Got: {dut.data_o.value}"

@cocotb.test
async def datapath_gate_test(dut):
    """Test for DatapathGate"""

    depth_p = 1
    l = depth_p
    rate = 1
    value = (1 << dut.InBits.value) - 1

    m = ModelRunner(dut, ElasticModel(dut))
    om = OutputModel(dut, RateGenerator(dut, 0), l)
    im = InputModel(dut, SingletonGenerator(dut, value), RateGenerator(dut, 0), l)

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
    im.start()

    # Wait... a bit.
    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)

    prior = dut.data_o
    await FallingEdge(dut.clk_i)
    dut.data_i.value = value
    await FallingEdge(dut.clk_i)

    if(dut.DatapathGate.value == 1):
        assert dut.data_o == prior,  f"Error! Value on data_o should not be updated when ready_o is high, when DatapathGate == 1. Expected: {prior.value}. Got: {dut.data_o.value}"
    else:
        assert dut.data_o == value,  f"Error! Value on data_o should not be updated when ready_o is high, when DatapathGate == 0. Expected: {value}. Got: {dut.data_o.value}"
import cocotb
from cocotb.triggers import RisingEdge, FallingEdge, with_timeout
from cocotb.result import SimTimeoutError

@cocotb.test
async def single_test(dut):
    """Test to transmit a single element in one cycle."""
    l = 1
    rate = 1

    m = ModelRunner(dut, ElasticModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), l)
    
    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    timeout_ns = 200
    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, f"Test timed out. Expected {l} outputs from {l} inputs."


@cocotb.test
async def fill_test(dut):
    """Test if module fills"""
    depth_p = 1
    l = depth_p

    m = ModelRunner(dut, ElasticModel(dut))
    om = OutputModel(dut, RateGenerator(dut, 0), l) # Block output
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, 1), l)

    dut.ready_i.value = 0
    dut.valid_i.value = 0    
    
    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    await RisingEdge(dut.valid_i)
    await RisingEdge(dut.clk_i)

    success = False
    try:
        # Give enough ns for the fifo to fill based on depth
        await im.wait((depth_p * 10) + 50) 
        success = True
    except SimTimeoutError:
        nconsumed = im.nconsumed()

    if not success:
        assert nconsumed != depth_p, f"Error! Could not fill fifo with {depth_p} elements. Consumed: {nconsumed}."


@cocotb.test
async def fill_empty_test(dut):
    """Test if module fills, then empties"""
    depth_p = 1
    l = depth_p

    m = ModelRunner(dut, ElasticModel(dut))
    om = OutputModel(dut, RateGenerator(dut, 0), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, 1), l)

    dut.ready_i.value = 0
    dut.valid_i.value = 0    
    
    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    await RisingEdge(dut.valid_i)
    await RisingEdge(dut.clk_i)

    # 1. Fill Phase
    success = False
    try:
        await im.wait((depth_p * 10) + 50)
        success = True
    except SimTimeoutError:
        nconsumed = im.nconsumed()

    if not success:
        assert nconsumed != depth_p, f"Error! Could not fill fifo. Consumed {nconsumed} elements."

    # 2. Empty Phase
    om.stop() # Stop the blocking output model
    om = OutputModel(dut, RateGenerator(dut, 1), l)
    om.start()

    await RisingEdge(dut.ready_i)
    await RisingEdge(dut.clk_i)

    nproduced = 0
    success = False
    try:
        await om.wait((depth_p * 10) + 50)
        success = True
    except SimTimeoutError:
        nproduced = om.nproduced()

    if not success:
        assert nproduced != depth_p, f"Error! Could not empty fifo. Produced {nproduced} elements."

async def rate_tests(dut, in_rate, out_rate):
    """Transmit 100 random data elements at 50% line rate (Output/Consumer is fuzzed)"""
    l_in = 100
    l_out = l_in

    m  = ModelRunner(dut, ElasticModel(dut))
    om = OutputModel(dut, RateGenerator(dut, out_rate), l_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, in_rate), l_in)

    dut.ready_i.value = 0
    dut.valid_i.value = 0    
    
    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    try:
        await with_timeout(RisingEdge(dut.valid_o), 20, 'ns')
    except SimTimeoutError:
        assert 0, "Test timed out waiting for valid_o to go high."

    slow = min(in_rate, out_rate)
    timeout_ns = int((l_out + 500) / max(slow, 0.05))

    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l_out} elements in {timeout_ns} ns. Transmitted: {om._nout}"

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
