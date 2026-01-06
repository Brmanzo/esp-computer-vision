import git
import os
import sys
import git
import queue
import math
import numpy as np
from typing import List
from functools import reduce

# I don't like this, but it's convenient.
_REPO_ROOT = git.Repo(search_parent_directories=True).working_tree_dir
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

from cocotb_test.simulator import run

from cocotbext.axi import AxiLiteBus, AxiLiteMaster, AxiStreamSink, AxiStreamMonitor, AxiStreamBus
   
import random
random.seed(42)

timescale = "1ps/1ps"

timescale = "1ps/1ps"
tests = ['reset_test'
         ,'single_test'
         ,'inout_fuzz_test'
         ,'in_fuzz_test'
         ,'out_fuzz_test'
         ,'full_bw_test'
         ,'full_bw_Gx_test'
         ,'full_bw_Gy_test']

def output_width(input_width: int, w_sum: int = 8) -> str:
    gray_max = (1 << input_width) - 1
    abs_w = math.ceil(math.log2(gray_max * w_sum + 1))
    return str(abs_w + 1)   # +1 for sign bit

def pack_weights(weights: List[int], width: int) -> int:
    # ws: list of 9 integers, each should fit in signed 3-bit (-4..3)
    # Pack weight 0 into bits [2:0], weight 1 into [5:3], etc.
    mask = (1 << width) - 1
    out = 0

    lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1

    for i, w in enumerate(weights):
        # convert signed to 3-bit two's complement
        assert lo <= w <= hi, f"weight[{i}]={w} out of range for signed {width}-bit ({lo}..{hi})"
        out |= (w & mask) << (width*i)
    return out

weights = [1]*9

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("linewidth_px_p, in_width_p, out_width_p", [("16", "2", output_width(8))])
def test_each(test_name, simulator, linewidth_px_p, in_width_p, out_width_p):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name)

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("linewidth_px_p, in_width_p, out_width_p", [("16", "2", output_width(2))])
def test_all(simulator, linewidth_px_p, in_width_p, out_width_p):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("linewidth_px_p, in_width_p, out_width_p", [("16", "2", output_width(2))])
def test_lint(simulator, linewidth_px_p, in_width_p, out_width_p):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("linewidth_px_p, in_width_p, out_width_p", [("16", "2", output_width(2))])
def test_style(simulator, linewidth_px_p, in_width_p, out_width_p):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

class SobelModel():
    def __init__(self, dut, weights: List[List[int]] = None):

        self._f = np.ones((3,3), dtype=int)
        self._dut = dut
        self._data_o = dut.data_o
        self._data_i = dut.data_i

        self._q = queue.SimpleQueue()
        
        self._linewidth_px_p = dut.linewidth_px_p.value

        # We're going to initialize _buf with NaN so that we can
        # detect when the output should be not an X in simulation
        self._buf = np.zeros((3,self._linewidth_px_p))/0
        self._deqs = 0
        self._enqs = 0

        # Surprise! You can change this.
        self.k = weights

    # Now let's scale this up a little bit
    # You can define functions to do the steps in convolution
    def update_window(self, buf, inp):
        temp = buf.flatten()

        # Now shift everything by 1
        temp = np.roll(temp, -1, axis=0)

        # Add the new input, replacing the input that was "kicked out"
        temp[-1] = inp

        # Now reshape it back into the original buffer
        temp = np.reshape(temp, buf.shape)
        buf = temp
        return buf

    def apply_kernel(self, buf):
        window = buf[:,-3:]
        # Now take the dot product between the window, and the kernel
        prod = np.multiply(self.k, window)
        result = prod.sum()
        return result

    def consume(self):
        assert_resolvable(self._data_i)
        self._q.put(self._data_i.value.integer)
        self._enqs += 1

    def produce(self):
        self._deqs += 1

        # Update the window model with the next input sample
        self._buf = self.update_window(self._buf, self._q.get())

        expected = self.apply_kernel(self._buf)

        if np.isnan(expected):
            return

        assert_resolvable(self._data_o)

        got = self._data_o.value.signed_integer

        expected = int(expected)
        assert got == expected, (
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

class AudioDataGenerator():
    def __init__(self, dut, l):
        self._dut = dut
        width = dut.in_width_p.value
        self._data = Audio.open(os.path.join(DIR_PATH, "image.jpg")).getdata()
        print(self._data[0])
        self._loc = 0

    def ninputs(self):
        return len(self._data)

    def generate(self):
        val = self._data[self._loc]
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
        data_i = self._dut.data_i

        await delay_cycles(self._dut, 1, False)

        if(not (reset_i.value.is_resolvable and reset_i.value == 0)):
            await FallingEdge(reset_i)

        await delay_cycles(self._dut, 2, False)

        # Precondition: Falling Edge of Clock
        din = self._data.generate()
        while self._nin < self._length:
            produce = self._rate.generate()
            success = 0
            valid_i.value = produce
            # Unsigned inputs should be zero extended as to not be misinterpreted by signed acc
            if int(self._dut.in_width_p.value) < 8:
                # only non-negative values that fit in signed w-bit (0 .. 2^(w-1)-1)
                data_i.value = din % (1 << (int(self._dut.in_width_p.value) - 1))
            else:
                data_i.value = din % (1 << int(self._dut.in_width_p.value))

            # Wait until ready
            while(produce and not success):
                await RisingEdge(clk_i)
                assert_resolvable(ready_o)
                #assert ready_o.value.is_resolvable, f"Unresolvable value in ready_o (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."

                success = True if (ready_o.value == 1) else False
                if (success):
                    din = self._data.generate()
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
            self._events.put(get_sim_time(units='ns'))
            self._model.consume()

    async def _run_output(self, model):
        while True:
            await self._rv_out.handshake(None)
            assert (self._events.qsize() > 0), "Error! Module produced output without valid input"
            input_time = self._events.get(get_sim_time(units='ns'))
            self._model.produce()
      
    def stop(self) -> None:
        """Stop monitor"""
        if self._coro_run is None:
            raise RuntimeError("Monitor never started")
        self._coro_run_input.kill()
        self._coro_run_output.kill()
        self._coro_run_input = None
        self._coro_run_output = None
    

@cocotb.test()
async def reset_test(dut):
    """Test for Initialization"""
    print("DUT objects:", dir(dut))
    clk_i = dut.clk_i
    reset_i = dut.reset_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)

@cocotb.test()
async def single_test(dut):
    """Test to transmit a single element in at most two cycles."""

    l = 1

    rate = 1
    kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]
    model = SobelModel(dut, kernel)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, 1), l)
    im = InputModel(dut, CountingDataGenerator(dut), RateGenerator(dut, rate), l)

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

@cocotb.test()
async def out_fuzz_test(dut):
    """Transmit data elements at 50% line rate (Output/Consumer is fuzzed)"""

    l = dut.linewidth_px_p.value * 4

    rate = .5

    timeout = 2 * l * int(1/rate)

    kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]
    model = SobelModel(dut, kernel)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, CountingGenerator(dut, rate), l)
    im = InputModel(dut, CountingDataGenerator(dut), RateGenerator(dut, 1), l)

    clk_i = dut.clk_i
    reset_i = dut.reset_i

    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    weights_i = dut.weights_i
    weights_i.value = pack_weights([1, 1, 1, 1, 1, 1, 1, 1, 1], 2)
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)

    # Wait one cycle for reset to start
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait for the first piece of data to arrive at the output.
    try:
        await with_timeout(RisingEdge(dut.valid_o), 20, 'ns')
    except cocotb.result.SimTimeoutError:
        assert 0, f"Test timed out. Testbench is waiting for valid_o, but valid_o never went high in 20 clock cycles after reset."

    try:
        await om.wait(timeout + .5)
    except cocotb.result.SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}. Only transmitted: {om._nout}"


@cocotb.test()
async def in_fuzz_test(dut):
    """Transmit data elements at 50% line rate (Input/Producer is fuzzed)"""

    l = dut.linewidth_px_p.value * 4
    rate = .5

    timeout = 2 * l * int(1/rate)

    kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]
    model = SobelModel(dut, kernel)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, 1), l)
    im = InputModel(dut, CountingDataGenerator(dut), CountingGenerator(dut, rate), l)

    clk_i = dut.clk_i
    reset_i = dut.reset_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    weights_i = dut.weights_i
    weights_i.value = pack_weights([1, 1, 1, 1, 1, 1, 1, 1, 1], 2)
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)

    # Wait one cycle for reset to start
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait for the first piece of data to arrive at the output.
    try:
        await with_timeout(RisingEdge(dut.valid_o), 20, 'ns')
    except cocotb.result.SimTimeoutError:
        assert 0, f"Test timed out. Testbench is waiting for valid_o, but valid_o never went high in 20 clock cycles after reset."

    try:
        await om.wait(timeout + .5)
    except cocotb.result.SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}. Only transmitted: {om._nout}"

@cocotb.test()
async def inout_fuzz_test(dut):
    """Transmit data elements at ~25% line rate (Both are fuzzed)"""

    l = dut.linewidth_px_p.value * 4
    rate = .5

    timeout = 2 * l * int(1/rate) * int(1/rate) 

    kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]
    model = SobelModel(dut, kernel)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, rate), l)
    im = InputModel(dut, CountingDataGenerator(dut), RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    reset_i = dut.reset_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0    
    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    weights_i = dut.weights_i
    weights_i.value = pack_weights([1, 1, 1, 1, 1, 1, 1, 1, 1], 2)
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)

    # Wait one cycle for reset to start
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait for the first piece of data to arrive at the output.
    try:
        await with_timeout(RisingEdge(dut.valid_o), 20, 'ns')
    except cocotb.result.SimTimeoutError:
        assert 0, f"Test timed out. Testbench is waiting for valid_o, but valid_o never went high in 20 clock cycles after reset."

    try:
        await om.wait(timeout + .5)
    except cocotb.result.SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}. Only transmitted: {om._nout}"
        
@cocotb.test()
async def full_bw_test(dut):
    """Transmit random data elements at 100% line rate"""

    l = dut.linewidth_px_p.value * 4
    rate = 1

    timeout = l + 1

    kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]
    model = SobelModel(dut, kernel)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, rate), l)
    im = InputModel(dut, CountingDataGenerator(dut), RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    reset_i = dut.reset_i

    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    weights_i = dut.weights_i
    weights_i.value = pack_weights([1, 1, 1, 1, 1, 1, 1, 1, 1], 2)

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)

    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # We're doing a throughput test. We only care about the output
    # throughput.  We can wait for the rising edge of valid_o because
    # it (should, if the circuit is implemented correctly) occur at,
    # or just after the clock edge.
    try:
        await with_timeout(RisingEdge(dut.valid_o), 20, 'ns')
    except cocotb.result.SimTimeoutError:
        assert 0, f"Test timed out. Testbench is waiting for valid_o, but valid_o never went high in 20 clock cycles after reset."

    try:
        await om.wait(timeout + .5)
    except cocotb.result.SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns"

@cocotb.test()
async def full_bw_Gx_test(dut):
    """Transmit random data elements at 100% line rate"""

    l = dut.linewidth_px_p.value * 4
    rate = 1

    timeout = l + 1

    kernel = [[-1, 0, 1],
              [-1, 0, 1],
              [-1, 0, 1]]
    model = SobelModel(dut, kernel)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, rate), l)
    im = InputModel(dut, CountingDataGenerator(dut), RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    reset_i = dut.reset_i

    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    weights_i = dut.weights_i
    weights_i.value = pack_weights([-1, 0, 1, -1, 0, 1, -1, 0, 1], 2)
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)

    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # We're doing a throughput test. We only care about the output
    # throughput.  We can wait for the rising edge of valid_o because
    # it (should, if the circuit is implemented correctly) occur at,
    # or just after the clock edge.
    try:
        await with_timeout(RisingEdge(dut.valid_o), 20, 'ns')
    except cocotb.result.SimTimeoutError:
        assert 0, f"Test timed out. Testbench is waiting for valid_o, but valid_o never went high in 20 clock cycles after reset."

    try:
        await om.wait(timeout + .5)
    except cocotb.result.SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns"

@cocotb.test()
async def full_bw_Gy_test(dut):
    """Transmit random data elements at 100% line rate"""

    l = dut.linewidth_px_p.value * 4
    rate = 1

    timeout = l + 1

    kernel = [[-1, -1, -1],
              [ 0,  0,  0],
              [ 1,  1,  1]]
    model = SobelModel(dut, kernel)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, rate), l)
    im = InputModel(dut, CountingDataGenerator(dut), RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    reset_i = dut.reset_i

    ready_i = dut.ready_i
    valid_i = dut.valid_i

    ready_i.value = 0
    valid_i.value = 0

    weights_i = dut.weights_i
    weights_i.value = pack_weights([-1, -1, -1, 0, 0, 0, 1, 1, 1], 2)
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, reset_i, 10)

    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # We're doing a throughput test. We only care about the output
    # throughput.  We can wait for the rising edge of valid_o because
    # it (should, if the circuit is implemented correctly) occur at,
    # or just after the clock edge.
    try:
        await with_timeout(RisingEdge(dut.valid_o), 20, 'ns')
    except cocotb.result.SimTimeoutError:
        assert 0, f"Test timed out. Testbench is waiting for valid_o, but valid_o never went high in 20 clock cycles after reset."

    try:
        await om.wait(timeout + .5)
    except cocotb.result.SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns"
