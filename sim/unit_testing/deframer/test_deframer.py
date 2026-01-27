# test_deframer.py
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

from cocotb.utils import get_sim_time
from cocotb.triggers import Timer, RisingEdge, FallingEdge, with_timeout
from cocotb.result import SimTimeoutError

   
import random
random.seed(42)

timescale = "1ps/1ps"

tests = ['reset_test', 'init_test', 'single_test', 'full_bw_test', 'fuzz_in_test', 'fuzz_out_test', 'fuzz_both_test', 'full_bw_repeat_test', 'fuzz_random_repeat_test']

def fxp_to_float(signal, frac):
    """Convert an unsigned fixed-point cocotb signal to a float."""
    return int(signal.value) / float(1 << frac)

def float_to_fxp(value, frac):
    """Convert a float to an unsigned fixed-point integer."""
    return int(round(value * (1 << frac)))

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("UnpackedWidth, PackedNum, PacketLenElems", [("2", "4", "10"), ("1", "8", "10")])
def test_each(test_name, simulator, UnpackedWidth, PackedNum, PacketLenElems):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_deframer")

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("UnpackedWidth, PackedNum, PacketLenElems", [("2", "4", "10"), ("1", "8", "10")])
def test_all(simulator, UnpackedWidth, PackedNum, PacketLenElems):
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

class DeframerModel():
    HEADER0 = 0
    HEADER1 = 1
    FORWARD = 2

    def __init__(self, dut):
        self._dut = dut
        self._unpacked_o = dut.unpacked_o
        self._data_i = dut.data_i
        self._UnpackedWidth  = int(dut.UnpackedWidth.value)
        self._PackedNum      = int(dut.PackedNum.value)
        self._PackedWidth    = int(dut.PackedWidth.value)
        self._PacketLenElems = int(dut.PacketLenElems.value)
        self._HeaderByte0    = int(dut.HeaderByte0.value)
        self._HeaderByte1    = int(dut.HeaderByte1.value)

        self._mask = (1 << self._UnpackedWidth) - 1
        self._step = 0
        self._packed_buf = None

        self._state = 0
        self._remaining = self._PacketLenElems * self._PackedNum
        
        self._deqs = 0
        self._enqs = 0

        self._q = queue.SimpleQueue()
    
    def consume(self):
        assert_resolvable(self._data_i)
        b = int(self._data_i.value) & ((1 << (self._PackedWidth)) - 1)
        # Detecting first byte
        if self._state == self.HEADER0:
            if b == self._HeaderByte0:
                self._state = self.HEADER1
                return False
            else:
                self._state = self.HEADER0
                return False
        # Detecting second byte after the first byte
        elif self._state == self.HEADER1:
            if b == self._HeaderByte1:
                self._state = self.FORWARD
                self._remaining = self._PacketLenElems * self._PackedNum
                return False
            elif b == self._HeaderByte0:
                # Stay in HEADER1 if we see another HEADER0
                self._state = self.HEADER1
                return False
            else:
                # Reset if unexpected byte
                self._state = self.HEADER0
        # Once both bytes detected, enqueue the rest of the packet
        elif self._state == self.FORWARD:
            if self._remaining >= self._PackedNum:
                self._q.put(b)
                self._enqs += 1
                return True
            else:
                # Packet complete: ignore until next header
                self._state = self.HEADER0
                return False

    def ensure_packed_buf(self):
        '''Checks for latest packed buffer state in queue before popping'''
        if self._packed_buf is None:
            assert (self._q.qsize() > 0), "Error! No input data available to pack"
            self._packed_buf = self._q.get()

    def _expected_from_byte(self, b: int, step: int) -> int:
        '''Masks the input byte to get the expected unpacked output for the given step'''
        return (b >> (self._UnpackedWidth * step)) & self._mask

    def produce(self):

        self.ensure_packed_buf()
        assert_resolvable(self._unpacked_o)

        got = int(self._unpacked_o.value) & self._mask
        if self._packed_buf is None:
            raise RuntimeError("Error! No packed buffer available to unpack")
        expected = self._expected_from_byte(self._packed_buf, self._step) & self._mask

        self._deqs += 1

        print(f'packed_byte: 0x{self._packed_buf:02X} step={self._step}')
        print(f'Got unpacked: {got}, Expected unpacked: {expected}')

        assert got == expected, (
            f"Mismatch on output #{self._deqs}: expected {expected}, got {got} "
            f"(packed=0x{self._packed_buf:02X}, step={self._step})"
        )
        if self._remaining > 0:
            self._remaining -= 1
        # Wrap around step
        self._step = (self._step + 1) % int(self._PackedNum)
        if self._step == 0:
            self._packed_buf = None
        
        if self._remaining == 0:
            self._state = self.HEADER0
            self._packed_buf = None
            self._step = 0

class ReadyValidInterface():
    def __init__(self, clk, reset, valid, ready):
        self._clk_i = clk
        self._rst_i = reset
        self._ready = ready
        self._valid = valid

    def is_in_reset(self):
        if (not self._rst_i.value.is_resolvable) or int(self._rst_i.value) == 1:
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

        # Countdown before header (random output)
        if self._header_delay > 0:
            self._header_delay -= 1
            return random.randint(0, mask)

        # Header byte 0
        if self._header_delay == 0:
            self._header_delay = -1
            return int(self._dut.HeaderByte0.value) & mask

        # Header byte 1
        if self._header_delay == -1:
            self._header_delay = -2
            self._period_remaining = self._period
            return int(self._dut.HeaderByte1.value) & mask

        # Payload period (random output)
        x_i = random.randint(0, mask)

        if self._period_remaining > 0:
            self._period_remaining -= 1

        if self._period_remaining == 0:
            self._repetitions -= 1
            if self._repetitions > 0:
                # wait initial_delay random bytes before next header sequence
                self._header_delay = self._initial_delay
                self._period_remaining = self._period  # pre-init; will be reset at -1->-2 anyway

        return x_i
    
class RandomDataGenerator():
    def __init__(self, dut):
        self._dut = dut
        self._width_p = dut.PackedWidth.value

    def generate(self):
        x_i = random.randint(0, (1 << self._width_p) - 1)
        return (x_i)
    
class EdgeCaseGenerator():

    def __init__(self, dut):
        self._dut = dut
        limits = [0, 1, (1 << self._dut.PackedWidth.value.integer) - 1]
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

        data = self._data.generate()

        # Precondition: Falling Edge of Clock
        while self._nin < self._length:
            produce = self._rate.generate()
            valid_i.value = produce
            data_i.value = data

            
            await RisingEdge(clk_i)
            assert_resolvable(ready_o)

            fire_in = (int(valid_i.value) == 1) and (int(ready_o.value) == 1)
            if(fire_in):
                self._nin += 1
                data = self._data.generate()

            await FallingEdge(clk_i)
            
        return self._nin

class ModelRunner():
    def __init__(self, dut, model):

        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._emit_cycles = int(dut.PackedNum.value)

        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.valid_i, dut.ready_o)
        self._rv_out = ReadyValidInterface(self._clk_i, self._rst_i,
                                           dut.valid_o, dut.ready_i)

        self._model = model

        self._events = queue.SimpleQueue()

        self._coro_run_in = None
        self._coro_run_out = None

    def start(self):
        if self._coro_run_in is not None:
            raise RuntimeError("Model already started")
        self._coro_run_in  = cocotb.start_soon(self._run_input(self._model))
        self._coro_run_out = cocotb.start_soon(self._run_output(self._model))

    async def _run_input(self, model):
        while True:
            await self._rv_in.handshake(None)
            forwarded = self._model.consume()
            if forwarded:
                for _ in range(self._emit_cycles): # Four valid outputs per valid input
                    self._events.put(get_sim_time(units='ns'))

    async def _run_output(self, model):
        while True:
            await self._rv_out.handshake(None)

            # Resolve same-cycle race vs _run_input()
            if self._events.qsize() == 0:
                # yield to let _run_input() run for this same timestep
                await Timer(Decimal(0), units="ns")

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
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

@cocotb.test
async def init_test(dut):
    """Test for Basic Connectivity"""

    clk_i = dut.clk_i
    rst_i = dut.rst_i

    dut.data_i.value = 0

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)


    await Timer(Decimal(1.0), units="ns")

    assert_resolvable(dut.unpacked_o)

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
        await om.wait(timeout + 10)
    except:
        timed_out = True
    assert not timed_out, "Error! Maximum latency expected for this fifo is two cycles."

    dut.valid_i.value = 0
    dut.ready_i.value = 0


@cocotb.test
async def full_bw_test(dut):
    """Input random data elements at 100% line rate"""
    delay = 10
    header_cycles = 2
    l = 10
    eg = RandomHeaderGenerator(dut, delay, repetitions=1, period=l)
    n_in = delay + header_cycles + l
    n_out = l * int(dut.PackedNum.value)
    rate = 1

    timeout = 20000

    m = ModelRunner(dut, DeframerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), n_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), n_in)

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

    try:
        await om.wait(timeout)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}"

@cocotb.test
async def fuzz_in_test(dut):
    """Add random data elements at 50% line rate"""
    delay = 10
    header_cycles = 2
    l = 10
    eg = RandomHeaderGenerator(dut, delay, repetitions=1, period=l)
    n_in = delay + header_cycles + l
    n_out = l * int(dut.PackedNum.value)
    rate = 0.5

    timeout = 20000

    m = ModelRunner(dut, DeframerModel(dut))
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
async def fuzz_out_test(dut):
    """Add random data elements at 50% line rate"""
    delay = 10
    header_cycles = 2
    l = 10
    eg = RandomHeaderGenerator(dut, delay, repetitions=1, period=l)
    n_in = delay + header_cycles + l
    n_out = l * int(dut.PackedNum.value)
    rate = 0.5

    timeout = 20000

    m = ModelRunner(dut, DeframerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), n_out)
    im = InputModel(dut, eg, RateGenerator(dut, 1), n_in)

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

    try:
        await om.wait(timeout)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}"

@cocotb.test
async def fuzz_both_test(dut):
    """Add random data elements at 50% line rate"""
    delay = 10
    header_cycles = 2
    l = 10
    eg = RandomHeaderGenerator(dut, delay, repetitions=1, period=l)
    n_in = delay + header_cycles + l
    n_out = l * int(dut.PackedNum.value)
    rate = 0.5

    timeout = 20000

    m = ModelRunner(dut, DeframerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), n_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), n_in)

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

    try:
        await om.wait(timeout)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}"


@cocotb.test
async def full_bw_repeat_test(dut):
    """Input random data elements at 100% line rate"""
    delay = 10
    header_cycles = 2
    repetitions = 3
    packet_len = 10

    eg = RandomHeaderGenerator(dut, delay, repetitions, period=packet_len)
    l = packet_len*repetitions
    n_in  = repetitions * (delay + header_cycles + packet_len)
    n_out = l * int(dut.PackedNum.value)
    
    rate = 1

    timeout = 20000

    m = ModelRunner(dut, DeframerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), n_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), n_in)

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

    try:
        await om.wait(timeout)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}"


@cocotb.test
async def fuzz_random_repeat_test(dut):
    """Add random data elements at 50% line rate"""
    delay = 10
    header_cycles = 2
    repetitions = 3
    packet_len = 10

    eg = RandomHeaderGenerator(dut, delay, repetitions, period=packet_len)
    l = packet_len*repetitions
    n_in  = repetitions * (delay + header_cycles + packet_len)
    n_out = l * int(dut.PackedNum.value)
    rate = 0.5

    timeout = 20000

    m = ModelRunner(dut, DeframerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), n_out)
    im = InputModel(dut, eg, RateGenerator(dut, rate), n_in)

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

    try:
        await om.wait(timeout)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l} elements in {timeout} ns, with output rate {rate}"