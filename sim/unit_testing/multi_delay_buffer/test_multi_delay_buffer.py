import git
import os
import sys
import git
import queue
import math
import numpy as np
from typing import List, Optional
from functools import reduce
from collections import deque

_REPO_ROOT = git.Repo(search_parent_directories=True).working_tree_dir
assert _REPO_ROOT is not None, "REPO_ROOT path must not be None"
assert (os.path.exists(_REPO_ROOT)), "REPO_ROOT path must exist"
_UTIL_PATH = os.path.join(_REPO_ROOT, "sim", "util")
assert os.path.exists(_UTIL_PATH), f"Utilities path does not exist: {_UTIL_PATH}"
sys.path.insert(0, _UTIL_PATH)
from utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence, delay_cycles
tbpath = os.path.dirname(os.path.realpath(__file__))

import pytest

import cocotb

from cocotb.triggers import RisingEdge, FallingEdge, with_timeout
from cocotb.result import SimTimeoutError

from cocotb_test.simulator import run
   
import random
random.seed(50)

timescale = "1ps/1ps"

timescale = "1ps/1ps"
tests = ['reset_test'
         ,'single_test'
         ,'inout_fuzz_test'
         ,'in_fuzz_test'
         ,'out_fuzz_test'
         ,'full_bw_test']

def safe_int_from_value(val, *, x_as=0):
    """
    Convert a cocotb BinaryValue/LogicArray to int even if it contains X/Z.
    x_as=0 -> treat X/Z as 0
    x_as=1 -> treat X/Z as 1
    """
    s = val.binstr.lower()  # e.g. '00x1'
    if 'x' in s or 'z' in s:
        repl = '1' if x_as else '0'
        s = s.replace('x', repl).replace('z', repl)
    return int(s, 2)

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("BufferWidth, Delay, BufferRows, InputChannels", [("1", "8", "2", "1"), ("1", "8", "2", "4"), ("2", "16", "4", "10")])
def test_each(test_name, simulator, BufferWidth, Delay, BufferRows, InputChannels):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name)

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("BufferWidth, Delay, BufferRows, InputChannels", [("1", "8", "2", "1"), ("1", "8", "2", "4"), ("2", "16", "4", "10")])
def test_all(simulator, BufferWidth, Delay, BufferRows, InputChannels):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("BufferWidth, Delay, BufferRows, InputChannels", [("1", "8", "2", "1")])
def test_lint(simulator, BufferWidth, Delay, BufferRows, InputChannels):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("BufferWidth, Delay, BufferRows, InputChannels", [("1", "8", "2", "1")])
def test_style(simulator, BufferWidth, Delay, BufferRows, InputChannels):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

def unpack_data_o(BufferWidth, BufferRows, InputChannels, packed_o):
    mask = (1 << BufferWidth) - 1
    out = [[0]*BufferRows for _ in range(InputChannels)]
    for ch in range(InputChannels):
        for r in range(BufferRows):
            bitpos = (ch * BufferRows + r) * BufferWidth
            out[ch][r] = (packed_o >> bitpos) & mask
    return out

class MultiDelayBufferModel():

    def __init__(self, dut):
        self._dut = dut
        self._data_i = dut.data_i
        self._data_o = dut.data_o

        self._q = queue.SimpleQueue()

        self._Delay         = int(dut.Delay.value)
        self._BufferRows    = int(dut.BufferRows.value)
        self._InputChannels = int(dut.InputChannels.value)
        self._BufferWidth   = int(dut.BufferWidth.value)

        self._mask = (1 << self._BufferWidth) - 1
        # Model the single cycle output delay
        self._warmup = self._Delay * self._BufferRows + 1

        # We're going to initialize _buf with zeros so that we can
        # detect when the output should be not an X in simulation
        self._deqs = 0
        self._enqs = 0

        # Deques representing vertical partitions within RAM
        zero_init = [0] * self._BufferRows

        self._ram = [
            deque([zero_init.copy() for _ in range(self._Delay)], maxlen=self._Delay)
            for _ in range(self._InputChannels)
        ]
        
        self._rd_pipe = None
        self._wr_pipe = [0 for _ in range(self._InputChannels)]  # delayed write data
        self._wr_valid = False

        self._fires = 0

        # Variables represent regs to provide single cycle delay between output
        # of buffer and input to the next buffer
        self._regs = [
            [0 for _ in range(self._BufferRows - 1)]
            for _ in range(self._InputChannels)
        ]

    def step(self, data_i_words, in_fire=True):
        assert len(data_i_words) == self._InputChannels

        # If not firing, return None (or handle as needed based on interface)
        if not in_fire:
            return None

        self._fires += 1

        rd_now = None  # word exiting delay line this cycle

        if self._wr_valid:
            rd_now = []

            for ch in range(self._InputChannels):
                # 1) Pop oldest word (this is what exits the delay line)
                old = self._ram[ch].popleft()
                rd_now.append(old.copy())

                # 2) Compute new row heads
                new_word0 = int(self._wr_pipe[ch]) & self._mask

                row_heads = [0] * self._BufferRows
                row_heads[0] = new_word0

                for r in range(1, self._BufferRows):
                    row_heads[r] = self._regs[ch][r - 1]

                # 3) Update inter-row regs from *current* old word
                for r in range(self._BufferRows - 1):
                    self._regs[ch][r] = old[r]

                # 4) Append new word to end of delay line
                self._ram[ch].append(row_heads)

        else:
            # Before first valid write, nothing meaningful leaves
            rd_now = [self._ram[ch][0].copy() for ch in range(self._InputChannels)]

        # 5) Update write pipeline
        self._wr_pipe = [int(w) & self._mask for w in data_i_words]
        self._wr_valid = True

        # 6) Output directly (Removed 1-cycle _rd_pipe delay)
        out = None
        
        # Adjusted threshold: Reduced (+2) to (+1) because we removed the pipeline stage.
        # If your hardware has 0-cycle output latency relative to the RAM read, 
        # you might need to adjust this constant further (e.g., to +0).
        latency_threshold = (self._Delay * self._BufferRows) + 1
        
        if self._fires >= latency_threshold:
            out = rd_now

        # Note: self._rd_pipe is removed entirely
        return out

class ReadyValidInterface():
    def __init__(self, clk, reset, valid, ready):
        self._clk_i = clk
        self._rst_i = reset
        self._ready = ready
        self._valid = valid

    def is_in_reset(self):
        return (not self._rst_i.value.is_resolvable) or (self._rst_i.value == 1)
    
    def assert_resolvable(self):
        if(not self.is_in_reset()):
            assert_resolvable(self._valid)
            assert_resolvable(self._ready)

    def is_handshake(self):
        return ((self._valid.value == 1) and (self._ready.value == 1))

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

class CountingDataGenerator():
    def __init__(self, dut):
        self._dut = dut
        self._cur = 0

    def generate(self):
        value = self._cur
        self._cur += 1
        return value
    
class RandomDataGenerator():
    def __init__(self, dut):
        self._dut = dut
        self._data = [0] * int(self._dut.InputChannels.value)
        self._first_high = False

    def generate(self):
        for ch in range(int(self._dut.InputChannels.value)):
            if not self._first_high:
                bw = int(self._dut.BufferWidth.value)
                self._data[ch] = (1 << bw) - 1
                self._first_high = True
            else:
                self._data[ch] = random.randint(0, (1 << int(self._dut.BufferWidth.value)) - 1)
        return self._data

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

                success = True if ready_i.value == 1 and valid_o.value == 1 else False
                if (success):
                    self._nout += 1

            await FallingEdge(clk_i)
        return self._nout

class InputModel():
    def pack_channels(self, words, w):
        mask = (1 << w) - 1
        packed = 0
        for ch, word in enumerate(words):
            packed |= (int(word) & mask) << (ch * w)
        return packed

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
        if self._coro is not None:
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
        din = self._data.generate()

        while self._nin < self._length:
            produce = self._rate.generate()
            valid_i.value = produce

            w = int(self._dut.BufferWidth.value)

            if produce:
                if w == 1:
                    # 1-bit data: allow 0/1
                    din_masked = [x & 0x1 for x in din]
                else:
                    # Keep MSB clear -> only non-negative signed values (0 .. 2^(w-1)-1)
                    din_masked = [x & ((1 << (w - 1)) - 1) for x in din]

                data_i.value = self.pack_channels(din_masked, w)

            success = False
            while produce and not success:
                await RisingEdge(clk_i)
                assert_resolvable(ready_o)
                success = bool(valid_i.value) and bool(ready_o.value)
                if success:
                    din = self._data.generate()
                    self._nin += 1

            await FallingEdge(clk_i)
        return self._nin
    
class ModelRunner():
    def unpack_data_i(self, packed_i: int):
        """Unpack packed [InputChannels-1:0][BufferWidth-1:0] into list of per-channel words."""
        C = int(self._dut.InputChannels.value)
        W = int(self._dut.BufferWidth.value)
        mask = (1 << W) - 1

        words = [0] * C
        for ch in range(C):
            words[ch] = (packed_i >> (ch * W)) & mask
        return words

    def __init__(self, dut, model):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut

        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.valid_i, dut.ready_o)
        self._rv_out = ReadyValidInterface(self._clk_i, self._rst_i,
                                           dut.valid_o, dut.ready_i)

        self._model = model
        self._events = queue.SimpleQueue()

        self._coro_run_in = None
        self._coro_run_out = None
        self._seen_expected = False

    def start(self):
        """Start model"""
        if self._coro_run_in is not None or self._coro_run_out is not None:
            raise RuntimeError("Model already started")
        self._coro_run_in  = cocotb.start_soon(self._run_input())
        self._coro_run_out = cocotb.start_soon(self._run_output())

    def stop(self):
        """Stop the model runner and kill background coroutines"""
        if self._coro_run_in is not None:
            self._coro_run_in.kill()
            self._coro_run_in = None
            
        if self._coro_run_out is not None:
            self._coro_run_out.kill()
            self._coro_run_out = None

    async def _run_input(self):
        while True:
            await self._rv_in.handshake(None)

            packed_in = int(self._dut.data_i.value)  # capture at handshake
            words = self.unpack_data_i(packed_in)    # ✅ list[int] length InputChannels
            expected = self._model.step(words, in_fire=True)
            if expected is not None:
                self._events.put(expected)
                self._seen_expected = True
    
    async def _run_output(self):
        while True:
            await self._rv_out.handshake(None)

            print(f"Output observed before expected: fire count {self._model._fires}, expected at least {self._model._warmup}")
            print(f"Input Channels: {self._model._InputChannels}, Buffer Rows: {self._model._BufferRows}, Delay: {self._model._Delay}")

            bw = self._model._BufferWidth
            mask = (1 << bw) - 1

            # Debug dump: model RAM vs DUT RAM
            for ch in range(self._model._InputChannels):
                for r in range(self._model._BufferRows):
                    ram_unpack = [0 for _ in range(self._model._Delay)]
                    data_o = self._model._dut.data_o.value

                    print(f"ch{ch}, br{r}:   ", end="")
                    for d in range(self._model._Delay):
                        print(f"{self._model._ram[ch][d][r]}", end="")
                    print("")

                    # show the corresponding output bit for that buffer row (your original view)
                    print(f"  mem     {data_o[self._model._BufferRows - r - 1]} ", end="")

                    # unpack DUT RAM taps
                    for d in range(self._model._Delay):
                        rowvec = safe_int_from_value(self._model._dut.ram_inst.mem[d].value, x_as=0)
                        word_idx = r + ch * self._model._BufferRows
                        dut_word = (rowvec >> (word_idx * bw)) & mask
                        ram_unpack[d] = dut_word

                    # rotate for visual alignment (your original rotate)
                    for d in range(len(ram_unpack)):
                        print(f"{ram_unpack[(d + self._model._fires - 1) % self._model._Delay]}", end="")
                    print("")
                print("")

            # ---- Scoreboard ----
            if self._events.qsize() == 0:
                # Ignore early output handshakes until we've ever produced an expected output
                # (ready/valid decoupling can allow this during warmup/drain)
                if not getattr(self, "_seen_expected", False):
                    continue

                raise AssertionError(
                    f"Output without expected input: fires={self._model._fires} "
                    f"warmup={self._model._warmup} qsize=0"
                )

            expected_words = self._events.get()

            val = self._dut.data_o.value
            if not val.is_resolvable:
                raise AssertionError(
                    f"data_o contains X/Z on output handshake at fires={self._model._fires}. "
                    f"data_o.binstr={val.binstr}"
                )

            got_packed = int(self._dut.data_o.value)
            got = unpack_data_o(
                self._dut.BufferWidth.value,
                self._dut.BufferRows.value,
                self._dut.InputChannels.value,
                got_packed
            )

            print(f"Output observed got={got} expected={expected_words}")
            assert got == expected_words, f"Mismatch: got={got} expected={expected_words}"

async def flush_dut(dut, duration):
    """
    Drives 0s into the DUT to overwrite any old data in the RAM.
    Does not check output, effectively ignoring 'garbage' from previous tests.
    """
    dut.valid_i.value = 1
    dut.ready_i.value = 1
    dut.data_i.value = 0 
    
    for _ in range(duration):
        await RisingEdge(dut.clk_i)
        
    dut.valid_i.value = 0
    dut.ready_i.value = 0

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""
    print("DUT objects:", dir(dut))
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

@cocotb.test
async def single_test(dut):

    D = int(dut.Delay.value)

    # Number of accepted inputs until first valid output position
    N_first = (dut.BufferRows.value) * D + 2

    # We expect exactly ONE output for this test
    N_out = 1

    rate = 1

    model = MultiDelayBufferModel(dut)
    m = ModelRunner(dut, model)

    om = OutputModel(dut, RateGenerator(dut, 1), N_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_first)

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    tmo_ns = 4 * N_first + 50
    timed_out = False

    try:
        await om.wait(tmo_ns)
    except SimTimeoutError:
        timed_out = True
    finally:

        try:
            im.stop()
        except Exception:
            pass

        try:
            om.stop()
        except Exception:
            pass

        try:
            m.stop()
        except Exception:
            pass

        # Drive interface to safe idle state
        dut.valid_i.value = 0
        dut.ready_i.value = 0

        # Give one cycle to settle before next test
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)

    assert not timed_out, (
        f"Timed out waiting for first valid output."
    )
@cocotb.test
async def out_fuzz_test(dut):

    D = int(dut.Delay.value)


    rate = 0.5

    model = MultiDelayBufferModel(dut)
    m = ModelRunner(dut, model)

    # Number of accepted inputs until first valid output position (x=K-1, y=K-1)
    N_in = (dut.BufferRows.value)*D*2 + 2 + model._warmup

    # We expect exactly ONE output for this test (the first valid position)
    N_out = N_in - model._warmup

    om = OutputModel(dut, RateGenerator(dut, rate), N_out)               # consume N_out outputs
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, 1), N_in)  # produce N_in inputs

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait until that single output is observed; timeout in ns but generous
    # If your clk is 1ns, N_first cycles is ~N_first ns; add cushion
    tmo_ns = 4 * N_in + 50

    timed_out = False
    try:
        await om.wait(tmo_ns)
    except SimTimeoutError:
        timed_out = True
    finally:

        try:
            im.stop()
        except Exception:
            pass

        try:
            om.stop()
        except Exception:
            pass

        try:
            m.stop()
        except Exception:
            pass

        # Drive interface to safe idle state
        dut.valid_i.value = 0
        dut.ready_i.value = 0

        # Give one cycle to settle before next test
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)

    assert not timed_out, (
        f"Timed out waiting for first valid output."
    )

@cocotb.test
async def in_fuzz_test(dut):

    D = int(dut.Delay.value)
    rows = int(dut.BufferRows.value)

    await clock_start_sequence(dut.clk_i)

    await reset_sequence(dut.clk_i, dut.rst_i, 10)


    flush_depth = rows * D + 5
    await flush_dut(dut, flush_depth)

    await reset_sequence(dut.clk_i, dut.rst_i, 10)


    await FallingEdge(dut.clk_i)

    rate = 0.5

    model = MultiDelayBufferModel(dut)
    m = ModelRunner(dut, model)

    # Number of accepted inputs until first valid output position (x=K-1, y=K-1)
    N_in = (dut.BufferRows.value)*D*2 + 2 + model._warmup

    # We expect exactly ONE output for this test (the first valid position)
    N_out = N_in - model._warmup

    om = OutputModel(dut, RateGenerator(dut, 1), N_out)               # consume N_out outputs
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_in)  # produce N_in inputs

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait until that single output is observed; timeout in ns but generous
    # If your clk is 1ns, N_first cycles is ~N_first ns; add cushion
    tmo_ns = 4 * N_in + 50

    timed_out = False
    try:
        await om.wait(tmo_ns)
    except SimTimeoutError:
        timed_out = True
    finally:

        try:
            im.stop()
        except Exception:
            pass

        try:
            om.stop()
        except Exception:
            pass

        try:
            m.stop()
        except Exception:
            pass

        # Drive interface to safe idle state
        dut.valid_i.value = 0
        dut.ready_i.value = 0

        # Give one cycle to settle before next test
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)

    assert not timed_out, (
        f"Timed out waiting for first valid output."
    )


@cocotb.test
async def inout_fuzz_test(dut):

    D = int(dut.Delay.value)

    rows = int(dut.BufferRows.value)

    await clock_start_sequence(dut.clk_i)

    await reset_sequence(dut.clk_i, dut.rst_i, 10)


    flush_depth = rows * D + 5
    await flush_dut(dut, flush_depth)

    await reset_sequence(dut.clk_i, dut.rst_i, 10)


    await FallingEdge(dut.clk_i)

    rate = 0.5

    model = MultiDelayBufferModel(dut)
    m = ModelRunner(dut, model)

    # Number of accepted inputs until first valid output position (x=K-1, y=K-1)
    N_in = (dut.BufferRows.value)*D*2 + 2 + model._warmup

    # We expect exactly ONE output for this test (the first valid position)
    N_out = N_in - model._warmup

    om = OutputModel(dut, RateGenerator(dut, rate), N_out)               # consume N_out outputs
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_in)  # produce N_in inputs

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait until that single output is observed; timeout in ns but generous
    # If your clk is 1ns, N_first cycles is ~N_first ns; add cushion
    tmo_ns = 4 * N_in + 50

    timed_out = False
    try:
        await om.wait(tmo_ns)
    except SimTimeoutError:
        timed_out = True
    finally:

        try:
            im.stop()
        except Exception:
            pass

        try:
            om.stop()
        except Exception:
            pass

        try:
            m.stop()
        except Exception:
            pass

        # Drive interface to safe idle state
        dut.valid_i.value = 0
        dut.ready_i.value = 0

        # Give one cycle to settle before next test
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)

    assert not timed_out, (
        f"Timed out waiting for first valid output."
    )
        
@cocotb.test
async def full_bw_test(dut):

    D = int(dut.Delay.value)

    rows = int(dut.BufferRows.value)

    await clock_start_sequence(dut.clk_i)

    await reset_sequence(dut.clk_i, dut.rst_i, 10)

    flush_depth = rows * D + 5
    await flush_dut(dut, flush_depth)

    await reset_sequence(dut.clk_i, dut.rst_i, 10)

    await FallingEdge(dut.clk_i)

    rate = 0.5

    model = MultiDelayBufferModel(dut)
    m = ModelRunner(dut, model)

    # Number of accepted inputs until first valid output position (x=K-1, y=K-1)
    N_in = (dut.BufferRows.value)*D*2 + 2 + model._warmup

    # We expect exactly ONE output for this test (the first valid position)
    N_out = N_in - model._warmup

    om = OutputModel(dut, RateGenerator(dut, 1), N_out)               # consume N_out outputs
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, 1), N_in)  # produce N_in inputs

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait until that single output is observed; timeout in ns but generous
    # If your clk is 1ns, N_first cycles is ~N_first ns; add cushion
    tmo_ns = 4 * N_in + 50

    timed_out = False
    try:
        await om.wait(tmo_ns)
    except SimTimeoutError:
        timed_out = True
    finally:

        try:
            im.stop()
        except Exception:
            pass

        try:
            om.stop()
        except Exception:
            pass

        try:
            m.stop()
        except Exception:
            pass

        # Drive interface to safe idle state
        dut.valid_i.value = 0
        dut.ready_i.value = 0

        # Give one cycle to settle before next test
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)

    assert not timed_out, (
        f"Timed out waiting for first valid output."
    )