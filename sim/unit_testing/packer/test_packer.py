# test_packer.py
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

tests = ['reset_test', 'init_test', 'single_test', 'full_bw_test', 'fuzz_random_test', 'flush_test']

def fxp_to_float(signal, frac):
    """Convert an unsigned fixed-point cocotb signal to a float."""
    return int(signal.value) / float(1 << frac)

def float_to_fxp(value, frac):
    """Convert a float to an unsigned fixed-point integer."""
    return int(round(value * (1 << frac)))

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("UnpackedWidth, PackedNum", [("2", "4"), ("1", "8")])
def test_each(test_name, simulator, UnpackedWidth, PackedNum):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_packer")

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("UnpackedWidth, PackedNum", [("2", "4"), ("1", "8")])
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
        self._dut              = dut
        self._unpacked_i       = dut.unpacked_i
        self._packed_o         = dut.packed_o
        self._UnpackedWidth = dut.UnpackedWidth.value
        self._PackedNum     = dut.PackedNum.value
        self._PackedWidth   = dut.PackedWidth.value
        self._flush_i          = dut.flush_i

        self._step         = 0
        self._acc          = 0
        
        self._deqs = 0
        self._enqs = 0

        self._q = queue.SimpleQueue()
    
    def consume(self):
        assert_resolvable(self._unpacked_i)
        u = int(self._unpacked_i.value) & ((1 << int(self._UnpackedWidth)) - 1)
        
        self._acc = self._acc | (u << (self._UnpackedWidth * self._step))
        self._enqs += 1

        assert_resolvable(self._flush_i)
        flush = (int(self._flush_i) == 1)

        completed = (self._step == self._PackedNum - 1) or flush
        if completed:
            self._q.put(self._acc & ((1 << self._PackedWidth) - 1))
            self._acc = 0
            self._step = 0
        else:
            self._step += 1
        return completed

    def produce(self):
        assert_resolvable(self._packed_o)

        got = self._packed_o.value.integer & ((1 << self._PackedWidth) - 1)

        assert self._q.qsize() > 0, (
            "Output fired but model has no completed expected byte. "
            "Did you call consume() on every input handshake?"
        )
        expected = self._q.get()
        self._deqs += 1

        print(f"Packed out #{self._deqs}: got=0x{got:02X}, expected=0x{expected:02X}")

        assert got == expected, (
            f"Mismatch on output #{self._deqs}: expected 0x{expected:02X}, got 0x{got:02X}"
        )

        if self._step == 0 or self._dut.flush_i.value == 1:
            self._packed_buf = None

class ReadyValidInterface():
    def __init__(self, clk, reset, valid, ready):
        self._clk_i = clk
        self._rst_i = reset
        self._ready = ready
        self._valid = valid

    def is_in_reset(self):
        if((not self._rst_i.value.is_resolvable) or self._rst_i.value  == 1):
            return True
        
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
    def __init__(self, dut, flush_rate):
        self._dut = dut
        self._width_p = dut.UnpackedWidth.value
        self._flush_rate = flush_rate

    def generate(self):
        if self._flush_rate == 0:
            x_i = random.randint(0, (1 << self._width_p) - 1)
            return (x_i)
        else:
            x_i = random.randint(0, (1 << self._width_p) - 1)
            if random.randint(1, int(1/self._flush_rate)) == 1:
                flush_i = 1
            else:
                flush_i = 0
            return (x_i, flush_i)
    
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
        self._rst_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.valid_i, dut.ready_o)

        self._rv_out = ReadyValidInterface(self._clk_i, self._rst_i,
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
        self._rst_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
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
        flush_i = self._dut.flush_i
        UnpackedWidth = self._dut.UnpackedWidth.value

        flush_i.value = 0

        await delay_cycles(self._dut, 1, False)

        if(not (rst_i.value.is_resolvable and rst_i.value == 0)):
            await FallingEdge(rst_i)

        await delay_cycles(self._dut, 2, False)

        def get_data():
            # Unpack generated data and flush values
            next_item = self._data.generate()
            if isinstance(next_item, tuple):
                data, flush = next_item
            else:
                data = next_item
                flush = 0

            # Mask data to width
            data = int(data) & ((1 << int(UnpackedWidth)) - 1)
            flush = int(flush) & 1
            return data, flush
        
        data, flush = get_data()

        # Precondition: Falling Edge of Clock
        while self._nin < self._length:
            produce = self._rate.generate()
            valid_i.value = produce
            unpacked_i.value = data
            flush_i.value = flush

            await RisingEdge(clk_i)
            assert_resolvable(ready_o)

            fire_in = (int(valid_i.value) == 1) and (int(ready_o.value) == 1)
            if(fire_in):
                self._nin += 1
                data, flush = get_data()

            await FallingEdge(clk_i)
            
        return self._nin

class ModelRunner():
    def __init__(self, dut, model):

        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._flush_i = dut.flush_i

        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.valid_i, dut.ready_o)
        self._rv_out = ReadyValidInterface(self._clk_i, self._rst_i,
                                           dut.valid_o, dut.ready_i)

        self._num_packed_p  = int(dut.PackedNum.value)

        self._model = model

        self._completed_packs = 0

        self._coro_run_in = None
        self._coro_run_out = None

    def start(self):
        """Start model"""
        if self._coro_run_in is not None:
            raise RuntimeError("Model already started")
        self._coro_run_input = cocotb.start_soon(self._run_input(self._model))
        self._coro_run_output = cocotb.start_soon(self._run_output(self._model))

    async def _run_input(self, model):
        while True:
            await self._rv_in.handshake(None)
            completed = self._model.consume()
            if completed:
                self._completed_packs += 1

    async def _run_output(self, model):
        while True:
            await self._rv_out.handshake(None)

            assert self._completed_packs > 0, (
                    f"Output fired with n_inputs={n} (<{self._num_packed_p}) "
                    f"but flush was not asserted (flush={int(self._flush_i.value)})."
                )
            self._completed_packs -= 1
            self._model.produce()
            
    def stop(self) -> None:
        """Stop monitor"""
        if self._coro_run_input is None or self._coro_run_output is None:
            raise RuntimeError("Monitor never started")
        self._coro_run_input.kill()
        self._coro_run_output.kill()
        self._coro_run_input = None
        self._coro_run_output = None

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
    dut.flush_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)


    await Timer(Decimal(1.0), units="ns")

    assert_resolvable(dut.packed_o)

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
    im = InputModel(dut, eg, RateGenerator(dut, rate), l_in)

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


@cocotb.test
async def full_bw_test(dut):
    """Input random data elements at 100% line rate"""

    eg = RandomDataGenerator(dut, flush_rate=0)
    l_out = 50
    l_in = l_out * dut.PackedNum.value
    rate = 1

    timeout = max(l_out, l_in) * int(1/rate) * int(1/rate) 

    m = ModelRunner(dut, PackerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), l_in)

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

@cocotb.test
async def fuzz_random_test(dut):
    """Add random data elements at 50% line rate"""

    eg = RandomDataGenerator(dut, flush_rate=0)
    l_out = 50
    l_in = l_out * dut.PackedNum.value
    rate = 0.5

    timeout = max(l_out, l_in) * int(1/rate) * int(1/rate) 

    m = ModelRunner(dut, PackerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), l_in)

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
    im = InputModel(dut, eg, RateGenerator(dut, rate), l_in)

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