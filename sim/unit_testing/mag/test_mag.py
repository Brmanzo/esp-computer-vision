# test_mag.py
import git
import os
import sys
import math
import numpy as np
import queue
from functools import reduce
from itertools import product
from decimal import Decimal

# I don't like this, but it's convenient.
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
from cocotb.types import LogicArray, Range
from cocotb.result import SimTimeoutError

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
def test_each(test_name, simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_mag")

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
def test_all(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, pymodule="test_mag")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, pymodule="test_mag")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"], pymodule="test_mag")


class MagModel():
    def __init__(self, dut):
        self._dut = dut
        self._mag_o = dut.mag_o
        self._gx_i = dut.gx_i
        self._gy_i = dut.gy_i

        self._deqs = 0
        self._enqs = 0

        self._tol = 0.05

        self._q = queue.SimpleQueue()
    
    def consume(self):
        assert_resolvable(self._gx_i)
        assert_resolvable(self._gy_i)
        self._q.put((self._gx_i.value, self._gy_i.value))
        self._enqs += 1

    def produce(self):
        self._deqs += 1

        gx = self._gx_i.value.integer
        gy = self._gy_i.value.integer

        assert_resolvable(self._mag_o)
        got = self._mag_o.value.integer
        if(gx > (2 * gy)):
            expected = gx
        elif(gy > (2 * gx)):
            expected = gy
        else:
            expected = int(gy + (gx/2))
        # expected = round((math.sqrt(self._gx_i.value.integer**2 + self._gy_i.value.integer**2)) * 2) / 2
        if np.isnan(expected):
            return

        print(f'gx_i: {self._gx_i.value.integer}, gy_i: {self._gy_i.value.integer}')
        print(f'Got magnitude: {got}, Expected magnitude: {expected}')
        assert abs((got > expected * (1 - self._tol)) or (got < expected * (1 + self._tol))), (
            f"Error! Output value on iteration {self._deqs} does not match expected. "
            f"Expected: {expected}. Got: {got}"
        )


class ReadyValidInterface():
    def __init__(self, clk, reset, ready, valid):
        self._clk_i = clk
        self._reset_i = reset
        self._ready = ready
        self._valid = valid

    def is_in_reset(self):
        if((not self._reset_i.value.is_resolvable) or self._reset_i.value  == 1):
            return True
        
    def assert_resolvable(self):
        if(not self.is_in_reset()):
            assert_resolvable(self._valid)
            assert_resolvable(self._ready)

    def is_handshake(self):
        return ((self._valid == 1) and (self._ready == 1))

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

    def generate(self):
        a_i = random.randint(0, (1 << self._dut.width_in_p.value) - 1)
        b_i = random.randint(0, (1 << self._dut.width_in_p.value) - 1)
        return (a_i, b_i)
    
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
        self._reset_i = dut.reset_i
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
        await with_timeout(self._coro, t, 'ns')

    def nproduced(self):
        return self._nout

    async def _run(self):
        """ Output Model Coroutine"""

        self._nout = 0
        clk_i = self._clk_i
        ready_i = self._dut.ready_i
        reset_i = self._dut.reset_i
        valid_o = self._dut.valid_o

        await FallingEdge(clk_i)

        if(not (reset_i.value.is_resolvable and reset_i.value == 0)):
            await FallingEdge(reset_i)

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
    def __init__(self, dut, data, rate, l):
        self._clk_i = dut.clk_i
        self._reset_i = dut.reset_i
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
        await with_timeout(self._coro, t, 'ns')

    def nconsumed(self):
        return self._nin

    async def _run(self):
        """ Input Model Coroutine"""

        self._nin = 0
        clk_i = self._clk_i
        reset_i = self._dut.reset_i
        ready_o = self._dut.ready_o
        valid_i = self._dut.valid_i
        gx_i = self._dut.gx_i
        gy_i = self._dut.gy_i

        await delay_cycles(self._dut, 1, False)

        if(not (reset_i.value.is_resolvable and reset_i.value == 0)):
            await FallingEdge(reset_i)

        await delay_cycles(self._dut, 2, False)

        data = self._data.generate()

        # Precondition: Falling Edge of Clock
        while self._nin < self._length:
            produce = self._rate.generate()
            success = 0
            valid_i.value = produce
            gx_i.value = data[0]
            gy_i.value = data[1]

            # Wait until ready
            while(produce and not success):
                await RisingEdge(clk_i)
                assert_resolvable(ready_o)
                #assert ready_o.value.is_resolvable, f"Unresolvable value in ready_o (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."

                success = True if (ready_o.value == 1) else False
                if (success):
                    self._nin += 1

            await FallingEdge(clk_i)
        return self._nin

class ModelRunner():
    def __init__(self, dut, model):

        self._clk_i = dut.clk_i
        self._reset_i = dut.reset_i

        self._rv_in = ReadyValidInterface(self._clk_i, self._reset_i,
                                          dut.valid_i, dut.ready_o)
        self._rv_out = ReadyValidInterface(self._clk_i, self._reset_i,
                                           dut.valid_o, dut.ready_i)

        self._model = model

        self._events = queue.SimpleQueue()

        self._coro_run_input = None
        self._coro_run_output = None

    def start(self):
        """Start model"""
        if self._coro_run_input is not None:
            raise RuntimeError("Model already started")
        self._coro_run_input = cocotb.start_soon(self._run_input(self._model))
        self._coro_run_output = cocotb.start_soon(self._run_output(self._model))

    async def _run_input(self, model):
        while True:
            await self._rv_in.handshake(None)
            self._events.put(get_sim_time(units='ns'))
            self._model.consume()

    async def _run_output(self, model):
        while True:
            await self._rv_out.handshake(None)
            assert (self._events.qsize() > 0), "Error! Module produced output without valid input"
            input_time = self._events.get()
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
    reset_i = dut.reset_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)

@cocotb.test
async def init_test(dut):
    """Test for Basic Connectivity"""

    clk_i = dut.clk_i
    reset_i = dut.reset_i

    dut.gx_i.value = 0
    dut.gy_i.value = 0
    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)


    await Timer(Decimal(1.0), units="ns")

    assert_resolvable(dut.mag_o)

@cocotb.test
async def single_test(dut):
    """Test to transmit a single element in at most two cycles."""

    l = 1
    eg = RandomDataGenerator(dut)

    rate = 1
   
    model = MagModel(dut)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, 1), l)
    im = InputModel(dut, eg, RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    reset_i = dut.reset_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)

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

    timeout = False
    try:
        await om.wait(2)
    except:
        timeout = True
    assert not timeout, "Error! Maximum latency expected for this circuit is one cycle."

    dut.valid_i.value = 0
    dut.ready_i.value = 0


@cocotb.test
async def full_bw_test(dut):
    """Input random data elements at 100% line rate"""

    eg = RandomDataGenerator(dut)
    l = 10
    rate = 1

    timeout = l + 4

    m = ModelRunner(dut, MagModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l)
    im = InputModel(dut, eg, RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    reset_i = dut.reset_i

    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)

    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    await RisingEdge(dut.ready_i)
    await RisingEdge(dut.clk_i)

    try:
        await om.wait(timeout)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}"


@cocotb.test
async def fuzz_random_test(dut):
    """Add random data elements at 50% line rate"""

    eg = RandomDataGenerator(dut)
    l = 10
    rate = 0.5

    timeout = l * int(1/rate) * int(1/rate) 

    m = ModelRunner(dut, MagModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l)
    im = InputModel(dut, eg, RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    reset_i = dut.reset_i

    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)

    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    await RisingEdge(dut.ready_i)
    await RisingEdge(dut.clk_i)

    try:
        await om.wait(timeout)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}"