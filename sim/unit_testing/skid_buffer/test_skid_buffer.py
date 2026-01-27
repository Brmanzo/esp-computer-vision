import git
import os
import sys
import git
from decimal import Decimal

_REPO_ROOT = git.Repo(search_parent_directories=True).working_tree_dir
assert _REPO_ROOT is not None, "REPO_ROOT path must not be None"
assert (os.path.exists(_REPO_ROOT)), "REPO_ROOT path must exist"
sys.path.append(os.path.join(_REPO_ROOT, "util"))
from utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence
tbpath = os.path.dirname(os.path.realpath(__file__))

import pytest

import cocotb

from cocotb.utils import get_sim_time
from cocotb.triggers import Timer,  RisingEdge, FallingEdge, with_timeout
from cocotb.result import SimTimeoutError
   
import random
random.seed(42)

import queue

timescale = "1ps/1ps"
tests = ['reset_test'
         ,'single_test'
         ,'inout_fuzz_test'
         ,'in_fuzz_test'
         ,'out_fuzz_test'
         ,'fill_test'
         ,'fill_empty_test'
         ,'full_bw_test'
         ]

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("Width, Depth, HeadRoom", [(8, 16, 4), (8, 32, 8)])  
def test_each(simulator, test_name, Width, Depth, HeadRoom):
    # retrieves simulators from simulator pytest param
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name)

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.

@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("Width, Depth, HeadRoom", [(8, 16, 4), (8, 32, 8)])
def test_all(simulator, Width, Depth, HeadRoom):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)


@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])


async def delay_cycles(dut, ncyc, polarity):
    for _ in range(ncyc):
        if(polarity):
            await RisingEdge(dut.clk_i)
        else:
            await FallingEdge(dut.clk_i)
    
class SkidBufModel():
    def __init__(self, dut):

        self._dut = dut
        self._data_o = dut.data_o
        self._data_i = dut.data_i

        # Model the fifo like a simple software queue
        self._q = queue.SimpleQueue()
        self._occupancy = 0
        
        self._Width = dut.Width.value
        self._Depth = dut.Depth.value
        self._HeadRoom = dut.HeadRoom.value
        self._deqs = 0
        self._enqs = 0

    def expect_rts(self):
        thresh = int(self._Depth) - int(self._HeadRoom)
        return self._occupancy >= thresh

    def consume(self):
        assert_resolvable(self._data_i)
        self._occupancy += 1
        self._q.put(self._data_i.value)
        self._enqs += 1

    def produce(self):
        assert_resolvable(self._data_o)
        self._occupancy -= 1
        got = self._data_o.value
        expected = self._q.get()
        assert got == expected, f"Error! Value on deque iteration {self._deqs} does not match expected. Expected: {expected}. Got: {got}"
        self._deqs += 1
        
        
class ReadyValidInterface():
    def __init__(self, clk, reset, ready, valid):
        self._clk_i = clk
        self._rst_i = reset
        self._ready = ready
        self._valid = valid

    def is_in_reset(self):
        if (not self._rst_i.value.is_resolvable or self._rst_i.value  == 1):
            return True
        return int(self._rst_i.value) == 1
        
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

    def generate(self):
        value = random.randint(0, (1 << self._dut.Width.value) - 1)
        return value

class RateGenerator():
    def __init__(self, dut, r):
        self._rate = r

    def generate(self):
        if(self._rate == 0):
            return False
        else:
            return (random.randint(1,int(1/self._rate)) == 1)

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


class OutputModel():
    def __init__(self, dut, g, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.ready_o, dut.valid_i)

        self._rv_out = ReadyValidInterface(self._clk_i, self._rst_i,
                                           dut.ready_i, dut.valid_o)
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

class InputModel():
    def __init__(self, dut, data, rate, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.ready_o, dut.valid_i)

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
        rst_i = self._dut.rst_i
        ready_o = self._dut.ready_o
        valid_i = self._dut.valid_i
        data_i = self._dut.data_i

        await delay_cycles(self._dut, 1, False)

        if(not (rst_i.value.is_resolvable and rst_i.value == 0)):
            await FallingEdge(rst_i)

        await delay_cycles(self._dut, 2, False)

        # Precondition: Falling Edge of Clock
        while self._nin < self._length:
            produce = self._rate.generate()
            din = self._data.generate()
            success = 0
            valid_i.value = produce
            data_i.value = din

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
        self._rst_i = dut.rst_i
        self._dut = dut

        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.ready_o, dut.valid_i)
        self._rv_out = ReadyValidInterface(self._clk_i, self._rst_i,
                                           dut.ready_i, dut.valid_o)

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
        self._coro_run_check = cocotb.start_soon(self._run_check(self._model))

    def dut_occ_modulo(self):
        pw = self._dut.write_ptr.value.n_bits  # clog2(depth)
        ptr_w = pw + 1
        mask = (1 << ptr_w) - 1

        ww = int(self._dut.write_wrap.value)
        wp = int(self._dut.write_ptr.value)
        rw = int(self._dut.read_wrap.value)
        rp = int(self._dut.read_ptr_q.value)

        w = ((ww << pw) | wp) & mask
        r = ((rw << pw) | rp) & mask
        occ = (w - r) & mask
        return occ, (ww, wp, rw, rp)

    async def _run_check(self, model):
        while True:
            await RisingEdge(self._clk_i)

            if not (self._rst_i.value.is_resolvable and int(self._rst_i.value) == 0):
                continue

            await Timer(Decimal(1.0), "ns")  # allow comb to settle after sequential updates

            exp = int(model.expect_rts())
            got = int(self._dut.rts_o.value)

            occ_hw, _ = self.dut_occ_modulo()
            thresh = int(self._dut.Depth.value) - int(self._dut.HeadRoom.value)
            exp = 1 if occ_hw >= thresh else 0
            got = int(self._dut.rts_o.value)
            assert got == exp, (f"rts mismatch @ {get_sim_time('ns')}ns: "
                f"occ_model={model._occupancy} occ_hw={occ_hw} "
                f"expected={exp} got={got}"
            )

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
        self._coro_run_check.kill()
        self._coro_run_input = None
        self._coro_run_output = None
    

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    Width = dut.Width.value

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

@cocotb.test
async def single_test(dut):
    """Test to transmit a single element in at most two cycles."""

    l = 1
    rate = 1

    m = ModelRunner(dut, SkidBufModel(dut))
    om = OutputModel(dut, RateGenerator(dut, 1), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i
    ready_o = dut.ready_o
    valid_o = dut.valid_o

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

    timeout = False
    try:
        await om.wait(3)
    except:
        timeout = True
    assert not timeout, "Error! Maximum latency expected for this fifo is two cycles."

    dut.valid_i.value = 0
    dut.ready_i.value = 0

@cocotb.test
async def bypass_test(dut):
    """Test to transmit a single element in one cycle."""

    l = 1
    rate = 1

    m = ModelRunner(dut, SkidBufModel(dut))
    om = OutputModel(dut, RateGenerator(dut, 1), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i
    ready_o = dut.ready_o
    valid_o = dut.valid_o

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

    timeout = False
    try:
        await om.wait(2)
    except:
        timeout = True
    assert not timeout, "Error! Maximum latency expected with bypass for this fifo is one cycle. " \
        "For maximum points (and minimum fifo latency), implement the FIFO bypass path."

    dut.valid_i.value = 0
    dut.ready_i.value = 0

@cocotb.test
async def fill_test(dut):
    """Test if fifo_1r1w fills to Depth elements"""

    Depth = dut.Depth.value
    l = Depth
    rate = 1

    m = ModelRunner(dut, SkidBufModel(dut))
    om = OutputModel(dut, RateGenerator(dut, 0), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i
    ready_o = dut.ready_o
    valid_o = dut.valid_o

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

    success = False
    try:
        await im.wait(Depth)
        success = True
    except:
        nconsumed = im.nconsumed()

    if(not success):
        assert nconsumed != Depth, f"Error! Could not fill fifo with {Depth} elements in {Depth} cycles. Fifo consumed {nconsumed} elements."
        
@cocotb.test
async def fill_empty_test(dut):
    """Test if fifo_1r1w fills to Depth elements"""

    Depth = dut.Depth.value
    l = Depth
    rate = 1

    m = ModelRunner(dut, SkidBufModel(dut))
    om = OutputModel(dut, RateGenerator(dut, 0), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i
    ready_o = dut.ready_o
    valid_o = dut.valid_o

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

    success = False
    try:
        await im.wait(Depth)
        success = True
    except:
        nconsumed = im.nconsumed()

    if(not success):
        assert nconsumed != Depth, f"Error! Could not fill fifo with {Depth} elements in {Depth} cycles. Fifo consumed {nconsumed} elements."

    om = OutputModel(dut, RateGenerator(dut, 1), l)
    om.start()

    await RisingEdge(dut.ready_i)
    await RisingEdge(dut.clk_i)

    nproduced = 0
    success = False
    try:
        await om.wait(Depth)
        success = True
    except:
        nproduced = om.nproduced()

    if(not success):
        assert nproduced != Depth, f"Error! Could not empty fifo with {Depth} elements in {Depth} cycles. Fifo produced {nproduced} elements."

@cocotb.test
async def out_fuzz_test(dut):
    """Transmit 4 * Depth random data elements at 50% line rate (Output/Consumer is fuzzed)"""

    l = dut.Depth.value * 4
    rate = .5

    timeout = 2 * l * int(1/rate)

    m = ModelRunner(dut, SkidBufModel(dut))
    om = OutputModel(dut, CountingGenerator(dut, rate), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, 1), l)

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i
    ready_o = dut.ready_o
    valid_o = dut.valid_o

    ready_i.value = 0
    valid_i.value = 0    
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    # Wait one cycle for reset to start
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait for the first piece of data to arrive at the output.
    try:
        await with_timeout(RisingEdge(dut.valid_o), 20, 'ns')
    except SimTimeoutError:
        assert 0, f"Test timed out. Testbench is waiting for valid_o, but valid_o never went high in 20 clock cycles after reset."

    #await RisingEdge(dut.valid_o)
    try:
        await om.wait(timeout + .5)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}. Only transmitted: {om._nout}"


@cocotb.test
async def in_fuzz_test(dut):
    """Transmit 4 * Depth random data elements at 50% line rate (Input/Producer is fuzzed)"""

    l = dut.Depth.value * 4
    rate = .5

    timeout = 2 * l * int(1/rate)

    m = ModelRunner(dut, SkidBufModel(dut))
    om = OutputModel(dut, RateGenerator(dut, 1), l)
    im = InputModel(dut, RandomDataGenerator(dut), CountingGenerator(dut, rate), l)

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i
    ready_o = dut.ready_o
    valid_o = dut.valid_o

    ready_i.value = 0
    valid_i.value = 0    
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    # Wait one cycle for reset to start
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait for the first piece of data to arrive at the output.
    try:
        await with_timeout(RisingEdge(dut.valid_o), 20, 'ns')
    except SimTimeoutError:
        assert 0, f"Test timed out. Testbench is waiting for valid_o, but valid_o never went high in 20 clock cycles after reset."
    #await RisingEdge(dut.valid_o)

    try:
        await om.wait(timeout + .5)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}. Only transmitted: {om._nout}"

@cocotb.test
async def inout_fuzz_test(dut):
    """Transmit 4 * Depth random data elements at ~25% line rate (Both are fuzzed)"""

    l = dut.Depth.value * 4
    rate = .5

    timeout = 2 * l * int(1/rate) * int(1/rate) 

    m = ModelRunner(dut, SkidBufModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i
    ready_o = dut.ready_o
    valid_o = dut.valid_o

    ready_i.value = 0
    valid_i.value = 0    
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    # Wait one cycle for reset to start
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait for the first piece of data to arrive at the output.
    try:
        await with_timeout(RisingEdge(dut.valid_o), 20, 'ns')
    except SimTimeoutError:
        assert 0, f"Test timed out. Testbench is waiting for valid_o, but valid_o never went high in 20 clock cycles after reset."

    #await RisingEdge(dut.valid_o)
    try:
        await om.wait(timeout + .5)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}. Only transmitted: {om._nout}"
        
@cocotb.test
async def full_bw_test(dut):
    """Transmit 8 * Depth random data elements at 100% line rate"""

    # This is the InputModel
    l = dut.Depth.value * 8
    rate = 1

    timeout = l + 1

    m = ModelRunner(dut, SkidBufModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), l)

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i
    ready_o = dut.ready_o
    valid_o = dut.valid_o

    ready_i.value = 0
    valid_i.value = 0    
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

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
    except SimTimeoutError:
        assert 0, f"Test timed out. Testbench is waiting for valid_o, but valid_o never went high in 20 clock cycles after reset."

    #await RisingEdge(dut.valid_o)
    try:
        await om.wait(timeout + .5)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns"
        