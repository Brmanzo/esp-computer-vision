# test_framer.py
from unittest import case
import git
import os
import sys
import git
import math
import numpy as np
import queue
from functools import reduce
from itertools import product

_REPO_ROOT = git.Repo(search_parent_directories=True).working_tree_dir
assert _REPO_ROOT is not None, "REPO_ROOT path must not be None"
assert (os.path.exists(_REPO_ROOT)), "REPO_ROOT path must exist"
sys.path.append(os.path.join(_REPO_ROOT, "util"))
from utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence, delay_cycles
tbpath = os.path.dirname(os.path.realpath(__file__))

import pytest

import cocotb

from cocotb.clock import Clock
from cocotb.regression import TestFactory
from cocotb.utils import get_sim_time
from cocotb.triggers import Timer, ClockCycles, RisingEdge, FallingEdge, with_timeout
from decimal import Decimal
from cocotb.types import LogicArray, Range

from cocotb_test.simulator import run

from cocotbext.axi import AxiLiteBus, AxiLiteMaster, AxiStreamSink, AxiStreamMonitor, AxiStreamBus
   
import random
random.seed(42)

timescale = "1ps/1ps"

tests = ['reset_test', 'init_test', 'single_test', 'full_bw_test', 'fuzz_random_test']

def fxp_to_float(signal, frac):
    """Convert an unsigned fixed-point cocotb signal to a float."""
    return int(signal.value) / float(1 << frac)

def float_to_fxp(value, frac):
    """Convert a float to an unsigned fixed-point integer."""
    return int(round(value * (1 << frac)))

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

        # Framer specific
        self._PacketLenElems = int(dut.PacketLenElems.value)
        self._tail0 = int(dut.TailByte0.value)
        self._tail1 = int(dut.TailByte1.value)

        # Packet counter
        self._count = 0

        # Packing State
        self._step  = 0
        self._acc   = 0
        
        self._deqs  = 0
        self._enqs  = 0

        self._q = queue.SimpleQueue()
    
    def consume(self):
        assert_resolvable(self._unpacked_i)
        u = int(self._unpacked_i.value) & ((1 << int(self._UnpackedWidth)) - 1)
        
        self._acc |= (u << (self._UnpackedWidth * self._step))
        self._enqs += 1

        last_elem = (self._count == (self._PacketLenElems - 1))

        # Packing behavior
        completed_pack = (self._step == self._PackedNum - 1) or last_elem
        if completed_pack:
            self._q.put(self._acc & ((1 << self._packed_width_p) - 1))
            self._acc = 0
            self._step = 0
        else:
            self._step += 1
        
        # If end of packet, enqueue the tail bytes
        if last_elem:
            self._q.put(self._tail0 & ((1 << self._packed_width_p) - 1))
            self._q.put(self._tail1 & ((1 << self._packed_width_p) - 1))
            self._count = 0
        else:
            self._count += 1

        if completed_pack and last_elem:
            return 3
        elif completed_pack:
            return 1
        else:
            return 0

    def produce(self):
        assert_resolvable(self._data_o)

        got = self._data_o.value.integer & ((1 << self._packed_width_p) - 1)

        assert self._q.qsize() > 0, (
            "Output fired but model has no completed expected byte. "
        )
        expected = self._q.get()
        self._deqs += 1

        assert got == expected, (
            f"Mismatch on output #{self._deqs}: expected 0x{expected:02X}, got 0x{got:02X}"
        )

class ReadyValidInterface():
    def __init__(self, clk, reset, valid, ready):
        self._clk_i = clk
        self._reset_i = reset
        self._ready = ready
        self._valid = valid

    def is_in_reset(self):
        if((not self._reset_i.value.is_resolvable) or self._reset_i.value  == 1):
            return True
        return False
        
    def assert_resolvable(self):
        if(not self.is_in_reset()):
            assert_resolvable(self._valid)
            assert_resolvable(self._ready)

    def is_handshake(self):
        return (int(self._valid.value) == 1) and (int(self._ready.value) == 1)

    async def _handshake(self):
        while True:
            await RisingEdge(self._clk_i)
            if (not self.is_in_reset()):
                self.assert_resolvable()
                if(self.is_handshake()):
                    break

    async def handshake(self, ns):
        """Wait for a handshake, raising an exception if it hasn't
        happened after ns nanoseconds of simulation time"""

        # If ns is none, wait indefinitely
        if(ns):
            await with_timeout(self._handshake(), ns, 'ns')
        else:
            await self._handshake()

class RandomDataGenerator():
    def __init__(self, dut):
        self._dut = dut
        self._width_p = dut.UnpackedWidth.value

    def generate(self):
        x_i = random.randint(0, (1 << self._width_p) - 1)
        return (x_i)
       
class EdgeCaseGenerator():

    def __init__(self, dut):
        self._dut = dut
        limits = [0, 1, (1 << self._dut.width_p.value) - 1]
        self._pairs = list(product(limits, limits))
        self._loc = 0

    def ninputs(self):
        return len(self._pairs)

    def generate(self):
        val = self._pairs[self._loc]
        self._loc += 1
        return val

class CountingDataGenerator():
    def __init__(self, dut):
        self._dut = dut
        self._cur = 0

    def generate(self):
        value = self._cur
        self._cur += 1
        return value

class CountingGenerator():
    def __init__(self, dut, r):
        self._rate = int(1/r)
        self._init = 0

    def generate(self):
        if(self._rate == 0):
            return False
        else:
            retval = (self._init == 1)
            self._init = (self._init + 1) % self._rate
            return retval

class RateGenerator():
    def __init__(self, dut, r):
        self._rate = r

    def generate(self):
        if(self._rate == 0):
            return False
        else:
            return (random.randint(1,int(1/self._rate)) == 1)

class OutputModel():
    def __init__(self, dut, g, l):
        self._clk_i = dut.clk_i
        self._reset_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._reset_i,
                                          dut.valid_i, dut.ready_o)

        self._rv_out = ReadyValidInterface(self._clk_i, self._reset_i,
                                           dut.valid_o, dut.ready_i)
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
        assert self._coro is not None
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

                fire_out = (int(valid_o.value) == 1) and (int(ready_i.value) == 1)
                if fire_out:
                    self._nout += 1
                    success = 1

            await FallingEdge(clk_i)
        return self._nout

class InputModel():
    def __init__(self, dut, data, rate, l):
        self._clk_i = dut.clk_i
        self._reset_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._reset_i,
                                          dut.valid_i, dut.ready_o)

        self._rate = rate
        self._data = data
        self._length = l

        self._coro = None

        self._nin = 0

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
        assert self._coro is not None
        await with_timeout(self._coro, t, 'ns')

    def nconsumed(self):
        return self._nin

    async def _run(self):
        """ Input Model Coroutine"""

        self._nin = 0
        clk_i = self._clk_i
        rst_i = self._dut.rst_i
        ready_o = self._dut.ready_o
        valid_i = self._dut.valid_i
        unpacked_i = self._dut.unpacked_i
        UnpackedWidth = self._dut.UnpackedWidth.value

        await delay_cycles(self._dut, 1, False)

        if(not (rst_i.value.is_resolvable and rst_i.value == 0)):
            await FallingEdge(rst_i)

        await delay_cycles(self._dut, 2, False)

        def get_data():
            # Unpack generated data and flush values
            # Mask data to width
            data = int(self._data.generate()) & ((1 << int(UnpackedWidth)) - 1)
            return data

        data = get_data()

        # Precondition: Falling Edge of Clock
        while self._nin < self._length:
            produce = self._rate.generate()
            valid_i.value = produce
            unpacked_i.value = data

            await RisingEdge(clk_i)
            assert_resolvable(ready_o)

            fire_in = (int(valid_i.value) == 1) and (int(ready_o.value) == 1)
            if(fire_in):
                self._nin += 1
                data = get_data()

            await FallingEdge(clk_i)
            
        return self._nin

class ModelRunner():
    def __init__(self, dut, model):
        self._dut = dut
        self._clk_i = dut.clk_i
        self._reset_i = dut.rst_i
        self._model = model
        self._expected_out_bytes = 0
        self._coro = None

    def start(self):
        if self._coro is not None:
            raise RuntimeError("ModelRunner already started")
        self._coro = cocotb.start_soon(self._run())

    def stop(self):
        if self._coro is None:
            raise RuntimeError("ModelRunner never started")
        self._coro.kill()
        self._coro = None

    async def _run(self):
        dut = self._dut

        while True:
            await RisingEdge(self._clk_i)

            # Skip reset cycles
            if (not self._reset_i.value.is_resolvable) or int(self._reset_i.value) == 1:
                self._expected_out_bytes = 0
                continue

            # INPUT handshake: update expected queue FIRST
            in_fire = (int(dut.valid_i.value) == 1) and (int(dut.ready_o.value) == 1)
            if in_fire:
                produced = self._model.consume()   # returns 0,1,3 (packed + optional tails)
                self._expected_out_bytes += int(produced)

            # OUTPUT handshake: then check/consume expected
            out_fire = (int(dut.valid_o.value) == 1) and (int(dut.ready_i.value) == 1)
            if out_fire:
                assert self._expected_out_bytes > 0, (
                    "Output fired but model expects no bytes. "
                    "This usually means the runner saw output before it accounted for input."
                )
                self._expected_out_bytes -= 1
                self._model.produce()

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
    im = InputModel(dut, eg, RateGenerator(dut, rate), l_in)

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

    timeout_cycles = int((l_in + l_out) * (1/rate) * dut.PackedNum.value) + 50
    timeout = False
    try:
        await om.wait(timeout_cycles)
    except:
        timeout = True
    assert not timeout, "Error! Maximum latency expected for this circuit is one cycle."

    dut.valid_i.value = 0
    dut.ready_i.value = 0


@cocotb.test
async def full_bw_test(dut):
    """Input random data elements at 100% line rate"""

    eg = RandomDataGenerator(dut)
    P = int(dut.PacketLenElems)
    K = int(dut.PackedNum)
    l_in, l_out = framer_lengths(K, P, num_packets=4)
    rate = 1

    m = ModelRunner(dut, FramerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), l_in)

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

    CLK_NS = 10
    timeout_cycles = int((l_in + l_out) * (1/rate) * dut.PackedNum.value) + 50
    timeout_ns = timeout_cycles * CLK_NS
    await om.wait(timeout_ns)

@cocotb.test
async def fuzz_random_test(dut):
    """Add random data elements at 50% line rate"""

    eg = RandomDataGenerator(dut)
    l_out = 50

    P = int(dut.PacketLenElems)
    K = int(dut.PackedNum)
    l_in, l_out = framer_lengths(K, P, num_packets=4)
    rate = 0.5

    m = ModelRunner(dut, FramerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), l_in)

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

    CLK_NS = 10
    timeout_cycles = int((l_in + l_out) * (1/rate) * dut.PackedNum.value) + 50
    timeout_ns = timeout_cycles * CLK_NS
    await om.wait(timeout_ns)