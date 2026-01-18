# test_rle_decode.py
from decimal import Decimal
import git
import os
import sys
import git
import math
import numpy as np
import queue
from functools import reduce
from itertools import product

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
   
from cocotb.triggers import RisingEdge, ReadOnly
import random
random.seed(42)

timescale = "1ps/1ps"

tests = ['reset_test', 'init_test', 'single_test', 'simple_test', 'full_bw_test', 'fuzz_random_test']

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
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_rle_decode")

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
def test_all(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, pymodule="test_rle_decode")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, pymodule="test_rle_decode")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"], pymodule="test_rle_decode")

class RLEDecodeModel:
    def __init__(self, dut):
        self._dut = dut
        self._data_o = dut.data_o
        self._rle_value_i = dut.rle_value_i
        self._rle_count_i = dut.rle_count_i

        self._cur_count = None
        self._cur_value = None
        self._step = 0  # how many beats already produced for current transaction

        self._deqs = 0
        self._enqs = 0
        self._q = queue.SimpleQueue()

    def consume(self):
        # Called on *input handshake*
        assert_resolvable(self._rle_count_i)
        assert_resolvable(self._rle_value_i)
        count = int(self._rle_count_i.value)
        value = int(self._rle_value_i.value)
        self._q.put((count, value))
        self._enqs += 1
        return count  # handy for ModelRunner bookkeeping

    def _ensure_cur(self):
        if self._cur_count is None:
            assert self._q.qsize() > 0, "Error! No input data available to decode"
            self._cur_count, self._cur_value = self._q.get()
            self._step = 0

    def produce(self):
        # Called on *output handshake*
        self._ensure_cur()
        assert_resolvable(self._data_o)

        got = int(self._data_o.value)
        expected = self._cur_value

        self._deqs += 1

        assert got == expected, (
            f"Mismatch on output #{self._deqs}: expected {expected}, got {got} "
            f"(count={self._cur_count}, step={self._step})"
        )

        # advance within current transaction
        self._step += 1
        if self._step is None or self._cur_count is None:
            raise RuntimeError("Internal error: step or cur_count is None")
        if self._step >= self._cur_count:
            # done with this run; next produce() will fetch next transaction
            self._cur_count = None
            self._cur_value = None
            self._step = 0

class ReadyValidInterface():
    def __init__(self, clk, reset, valid, ready):
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
        value_i = random.randint(0, (1 << self._dut.data_width_p.value) - 1)
        count_i = random.randint(0, (1 << self._dut.count_width_p.value) - 1)
        return (value_i, count_i)

class SingleRLEGen:
    def __init__(self, value, count):
        self.value = value
        self.count = count
        self.sent = False

    def generate(self):
        # return the same transaction each time; or raise after first if you prefer
        return (self.value, self.count)
    
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
        
class FixedDataGenerator():
    def __init__(self, dut, count_val=None):
        self._dut = dut
        # Default to max count if not specified
        if count_val is None:
            self._count_val = (1 << self._dut.count_width_p.value) - 1
        else:
            self._count_val = count_val

    def generate(self):
        value_i = random.randint(0, (1 << self._dut.data_width_p.value) - 1)
        count_i = self._count_val
        return (value_i, count_i)
    
class PrecomputedListGenerator:
    """Feeds a pre-calculated list of (value, count) tuples to the InputModel"""
    def __init__(self, data_list):
        self._data = data_list
        self._iter = iter(data_list)
    
    def generate(self):
        return next(self._iter)

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
            while consume and not success:
                await RisingEdge(clk_i)
                assert_resolvable(valid_o)

                fire_out = (valid_o.value.is_resolvable and int(valid_o.value) == 1
                            and ready_i.value.is_resolvable and int(ready_i.value) == 1)

                if fire_out:
                    self._nout += 1
                    success = 1

            await FallingEdge(clk_i)
        return self._nout

class InputModel():
    def __init__(self, dut, data, rate, l):
        self._clk_i = dut.clk_i
        self._reset_i = dut.reset_i
        self._dut = dut

        self._rv_in = ReadyValidInterface(self._clk_i, self._reset_i,
                                          dut.valid_i, dut.ready_o)

        self._rate = rate          # decides when to START a new txn
        self._data = data          # provides (value, count)
        self._length = l

        self._coro = None
        self._nin = 0

    def start(self):
        if self._coro is not None:
            raise RuntimeError("Input Model already started")
        self._coro = cocotb.start_soon(self._run())

    def stop(self) -> None:
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
        self._nin = 0
        clk_i = self._clk_i
        reset_i = self._dut.reset_i
        ready_o = self._dut.ready_o
        valid_i = self._dut.valid_i
        rle_value_i = self._dut.rle_value_i
        rle_count_i = self._dut.rle_count_i

        # settle
        await delay_cycles(self._dut, 1, False)
        if not (reset_i.value.is_resolvable and int(reset_i.value) == 0):
            await FallingEdge(reset_i)
        await delay_cycles(self._dut, 2, False)

        # pending transaction state
        pending = False
        cur_value = 0
        cur_count = 0

        # Precondition: Falling Edge of Clock
        while self._nin < self._length:
            # If we don't currently have a txn in-flight, maybe start one
            if not pending:
                start = self._rate.generate()
                if start:
                    cur_value, cur_count = self._data.generate()
                    pending = True
                else:
                    valid_i.value = 0

            # Drive signals
            if pending:
                valid_i.value = 1
                rle_value_i.value = cur_value
                rle_count_i.value = cur_count

            await RisingEdge(clk_i)
            assert_resolvable(ready_o)

            fire_in = (valid_i.is_resolvable and int(valid_i.value) == 1 and
                       ready_o.is_resolvable and int(ready_o.value) == 1)

            if fire_in:
                # transaction accepted this cycle
                self._nin += 1
                pending = False
                # (optional) drop valid next cycle; we'll do it naturally when pending=False

            await FallingEdge(clk_i)

        # clean up drives (optional)
        valid_i.value = 0
        return self._nin
    
class ModelRunner():
    def __init__(self, dut, model):

        self._clk_i = dut.clk_i
        self._reset_i = dut.reset_i
        self._count_i = dut.rle_count_i

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
            count = model.consume()          # <--- enqueue txn + get count
            for _ in range(count):
                self._events.put(get_sim_time(units='ns'))

    async def _run_output(self, model):
        while True:
            await self._rv_out.handshake(None)
            assert (self._events.qsize() > 0), "Error! Module produced output without valid input"
            _ = self._events.get()
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

    dut.rle_value_i.value = 0
    dut.rle_count_i.value = 1

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)


    await Timer(Decimal(1.0), units="ns")

    assert_resolvable(dut.data_o)

@cocotb.test
async def single_test(dut):
    """Test to transmit a single element in at most two cycles."""

    l  = 1
    eg = SingleRLEGen(value=1, count=1)

    rate = 1
   
    model = RLEDecodeModel(dut)
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
        await om.wait(64)
    except:
        timeout = True
    assert not timeout, "Error! Maximum latency expected for this circuit is one cycle."

    dut.valid_i.value = 0
    dut.ready_i.value = 0

@cocotb.test
async def simple_test(dut):

    values = [1, 3, 1, 2, 0]
    counts = [13, 2, 7, 5, 20]

    clk_i = dut.clk_i
    reset_i = dut.reset_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i

    count_i = dut.rle_count_i
    value_i = dut.rle_value_i

    ready_i.value = 0
    valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)

    # Wait one cycle for reset to start
    ready_i.value = 1
    valid_i.value = 1
    time = 0

    timeout = sum(counts) * 10
    for v, c in zip(values, counts):
        value_i.value = v
        count_i.value = c
        for _ in range(c):
            if time < timeout:
                await RisingEdge(dut.clk_i)
                time += 1
            else:
                assert 0, "Error! Expected output valid after all inputs sent."

@cocotb.test
async def full_bw_test(dut):
    """Input data elements with MAX count at 100% line rate"""
    
    # 1. Use the Fixed Generator (keeps input/output counts identical)
    eg = FixedDataGenerator(dut) 
    
    l = 10
    rate = 1
    max_count = (1 << int(dut.count_width_p.value)) - 1
    
    # 2. Correct the expectation math
    expected_outputs = l * max_count

    m = ModelRunner(dut, RLEDecodeModel(dut))
    
    # 3. Pass expected_outputs
    om = OutputModel(dut, RateGenerator(dut, rate), expected_outputs)
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
    om.start() # OutputModel sets ready_i=1 almost immediately

    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    
    im.start()

    # REMOVED: await RisingEdge(dut.ready_i) 
    # Because ready_i is already 1, this would hang forever.

    await RisingEdge(dut.clk_i) # Just sync to clock once if needed

    timeout = expected_outputs + 500

    try:
        await om.wait(timeout)
    except SimTimeoutError:
        assert 0, f"Timed out: expected {expected_outputs} outputs in {timeout} cycles"

    dut.valid_i.value = 0
    dut.ready_i.value = 0

@cocotb.test
async def fuzz_random_test(dut):
    """Add random data elements at 50% line rate with PRE-CALCULATED totals"""

    l = 10
    rate = 0.5
    
    # Pre-calculate the Random Data
    data_width = int(dut.data_width_p.value)
    count_width = int(dut.count_width_p.value)
    max_val = (1 << data_width) - 1
    max_cnt = (1 << count_width) - 1
    
    txn_list = []
    total_expected_outputs = 0
    
    for _ in range(l):
        v = random.randint(0, max_val)
        c = random.randint(0, max_cnt)
        txn_list.append((v, c))
        total_expected_outputs += c

    # Setup Generators with known data
    eg = PrecomputedListGenerator(txn_list)

    # InputModel gets length 'l' (number of transactions)
    # OutputModel gets 'total_expected_outputs' (sum of all counts)

    timeout = (total_expected_outputs * (1/rate)) + 500

    m = ModelRunner(dut, RLEDecodeModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), total_expected_outputs)
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
    
    try:
        await om.wait(timeout)
    except SimTimeoutError:
        assert 0, (f"Test timed out. Expected {total_expected_outputs} outputs "
                   f"from {l} inputs.")

    # Cleanup
    dut.valid_i.value = 0
    dut.ready_i.value = 0
