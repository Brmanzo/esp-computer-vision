# test_framer.py
from   pathlib import Path
import pytest
import queue

from util.utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from util.gen_inputs import gen_random_unsigned
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import Decimal, Timer, RisingEdge, FallingEdge
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
@pytest.mark.parametrize("UnpackedWidth, PackedNum, PacketLenElems", [("2", "4", "124"), ("1", "8", "124")])
def test_each(test_name, simulator, UnpackedWidth, PackedNum, PacketLenElems):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_framer")

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("UnpackedWidth, PackedNum, PacketLenElems", [("2", "4", "124"), ("1", "8", "124")])
def test_all(simulator, UnpackedWidth, PackedNum, PacketLenElems):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, pymodule="test_framer")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, pymodule="test_framer")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"], pymodule="test_framer")

class FramerModel():
    def __init__(self, dut):
        self._dut                = dut
        self._unpacked_i         = dut.unpacked_i
        self._data_o             = dut.data_o

        self._UnpackedWidth   = int(dut.UnpackedWidth.value)
        self._PackedNum       = int(dut.PackedNum.value)
        self._packed_width_p     = int(dut.PackedWidth.value)

        self._PacketLenElems = int(dut.PacketLenElems.value)
        self._WakeupCmd = int(dut.WakeupCmd.value)
        self._tail0 = int(dut.TailByte0.value)
        self._tail1 = int(dut.TailByte1.value)

        self._count = 0
        self._step  = 0
        self._acc   = 0
        self._deqs  = 0

    def reset(self):
        # Clear model state to match DUT reset behavior
        self._count = 0
        self._wakeup_sent = False
        self._step = 0
        self._acc = 0

        # reset expected queue
        self._q = queue.SimpleQueue()

        # Wakeup is emitted once after reset release
        self._q.put(self._WakeupCmd)
        self._wakeup_sent = True
    
    def consume(self):
        assert_resolvable(self._unpacked_i)
        u = int(self._unpacked_i.value) & ((1 << int(self._UnpackedWidth)) - 1)
        
        # Shift in new unpacked byte to accumulator
        self._acc |= (u << (self._UnpackedWidth * self._step))

        last_elem = (self._count == (self._PacketLenElems - 1))
        completed_pack = (self._step == self._PackedNum - 1) or last_elem

        expected_outputs = []

        # Pack completed bytes
        if completed_pack:
            expected_outputs.append(self._acc & ((1 << self._packed_width_p) - 1))
            self._acc = 0
            self._step = 0
        else:
            self._step += 1
        
        # If end of packet, enqueue the tail bytes
        if last_elem:
            expected_outputs.append(self._tail0 & ((1 << self._packed_width_p) - 1))
            expected_outputs.append(self._tail1 & ((1 << self._packed_width_p) - 1))
            self._count = 0
        else:
            self._count += 1

        # Return None, single value, or list!
        if len(expected_outputs) == 0:
            return None
        elif len(expected_outputs) == 1:
            return expected_outputs[0]
        else:
            return expected_outputs

    def produce(self, expected):
        assert_resolvable(self._data_o)

        got = self._data_o.value.integer & ((1 << self._packed_width_p) - 1)
        self._deqs += 1

        print(f"Output #{self._deqs}: Expected 0x{expected:02X}, Got 0x{got:02X}")
        assert got == expected, (
            f"Mismatch on output #{self._deqs}: expected 0x{expected:02X}, got 0x{got:02X}"
        )

class RandomDataGenerator():
    def __init__(self, dut):
        self._dut = dut
        self._width_p = dut.UnpackedWidth.value

    def generate(self):
        val = gen_random_unsigned(self._width_p, rng=random)
        return val, val

def framer_lengths(PackedNum: int, PacketLenElems: int, num_packets: int):
    P = PacketLenElems
    K = PackedNum
    l_in = num_packets * P
    packed_per_packet = (P + K - 1) // K
    l_out = num_packets * (packed_per_packet + 2)
    return l_in, l_out

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

@cocotb.test
async def init_test(dut):
    """Test for Basic Connectivity"""

    clk_i = dut.clk_i
    rst_i = dut.rst_i

    dut.unpacked_i.value = 0

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)


    await Timer(Decimal(1.0), units="ns")

    assert_resolvable(dut.data_o)

@cocotb.test
async def single_test(dut):
    """Test to transmit a single element in at most two cycles."""

    eg = RandomDataGenerator(dut)
    l_out = 1
    l_in = l_out * dut.PackedNum.value
    rate = 1

    timeout = max(l_out, l_in) * int(1/rate) * int(1/rate) 

    m = ModelRunner(dut, FramerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), l_in, data_pins=dut.unpacked_i)

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)
    await FallingEdge(dut.clk_i)

    m._events.put(int(dut.WakeupCmd.value))

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

async def rate_tests(dut, in_rate, out_rate):
    eg = RandomDataGenerator(dut)
    P = int(dut.PacketLenElems)
    K = int(dut.PackedNum)
    l_in, l_out = framer_lengths(K, P, num_packets=4)

    m = ModelRunner(dut, FramerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, in_rate), l_out)
    im = InputModel(dut, eg, RateGenerator(dut, out_rate), l_in, data_pins=dut.unpacked_i)

    clk_i = dut.clk_i
    rst_i = dut.rst_i

    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)
    await FallingEdge(dut.clk_i)

    m._events.put(int(dut.WakeupCmd.value))

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
            f"Timed out. Expected {l_out} output handshakes "
            f"(N={l_out}). Got {om.nproduced()} in {timeout_ns} ns. "
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