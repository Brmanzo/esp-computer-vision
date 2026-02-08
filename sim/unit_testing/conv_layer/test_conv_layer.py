import git
import os
import sys
import git
import queue
import math
import numpy as np
from typing import List, Optional
from functools import reduce

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
         ,'full_bw_test'
         ,'full_bw_Gxy_test']

def output_width(input_width: int, w_sum: int = 8) -> str:
    '''Calculates proper output width for given input width amount of accumulations.'''
    gray_max = (1 << input_width) - 1
    abs_w = math.ceil(math.log2(gray_max * w_sum + 1))
    return str(abs_w + 1)   # +1 for sign bit

def pack_weights_channels(weights_by_ch, width, kernel_area):
    mask = (1 << width) - 1
    lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
    out = 0
    for ch, ws in enumerate(weights_by_ch):
        assert len(ws) == kernel_area
        for k, w in enumerate(ws):
            assert lo <= w <= hi
            shift = width * (k + kernel_area * ch)   # k first, then channel
            out |= (w & mask) << shift
    return out

def sign_extend(val: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (val ^ sign) - sign

def flatten_kernel(k2d):
    # row-major flatten: k[r][c] -> k[r*K+c]
    return [v for row in k2d for v in row]

weights = [1]*9

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("LineWidthPx, WidthIn, WidthOut, KernelWidth, OutChannels, Stride", 
                         [("16", "1", output_width(1), "3", "1", "1"),
                          ("32", "2", output_width(2), "5", "1", "1"),
                          ("16", "1", output_width(1), "3", "2", "1"),
                          ("16", "1", output_width(1), "3", "1", "2")])
def test_each(test_name, simulator, LineWidthPx, WidthIn, WidthOut, KernelWidth, OutChannels, Stride):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name)

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("LineWidthPx, WidthIn, WidthOut, KernelWidth, OutChannels, Stride", 
                         [("16", "1", output_width(1), "3", "1", "1"),
                          ("32", "2", output_width(2), "5", "1", "1"),
                          ("16", "1", output_width(1), "3", "2", "1")])
def test_all(simulator, LineWidthPx, WidthIn, WidthOut, KernelWidth, OutChannels, Stride):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("LineWidthPx, WidthIn, WidthOut", [("16", "2", output_width(2))])
def test_lint(simulator, LineWidthPx, WidthIn, WidthOut):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("LineWidthPx, WidthIn, WidthOut", [("16", "2", output_width(2))])
def test_style(simulator, LineWidthPx, WidthIn, WidthOut):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

class ConvLayerModel():
    def __init__(self, dut, weights: List[List[List[int]]], input_height: int):
        self._kernel_width = int(dut.KernelWidth.value)
        self._f = np.ones((self._kernel_width,self._kernel_width), dtype=int)
        self._dut = dut
        self._data_o = dut.data_o
        self._data_i = dut.data_i

        self._q = queue.SimpleQueue()

        self._input_width = int(dut.LineWidthPx.value)
        self._input_height = input_height
        self._WidthOut = int(dut.WidthOut.value)
        self._channels = int(dut.OutChannels.value)

        self._Origin = int(dut.Origin.value)
        self._Stride = int(dut.Stride.value)

        # We're going to initialize _buf with NaN so that we can
        # detect when the output should be not an X in simulation
        self._buf = np.zeros((self._kernel_width,self._input_width))/0
        self._deqs = 0
        self._enqs = 0

        self.k = np.array(weights, dtype=int)
        assert self.k.shape == (self._channels, self._kernel_width, self._kernel_width)

        self._in_idx = 0

        # Valid kernel positions within input image depending on stride and kernel size
        invalid_region = self._kernel_width - 1
        S = int(self._Stride)
        O = int(self._Origin)
        if S <= 1: 
            S = 1
            O = 0
        
        self._valid_cycles = np.ones((self._input_height, self._input_width), dtype=bool)
        self._valid_cycles[:invalid_region, :] = False
        self._valid_cycles[:, :invalid_region] = False
        
        if S > 1:
            for r in range(invalid_region, self._input_height):
                for c in range(invalid_region, self._input_width):
                    if ((r - invalid_region) % S) != O or ((c - invalid_region) % S) != O:
                        self._valid_cycles[r, c] = False

    def _produces_output(self, idx: int) -> bool:
        x = idx % int(self._input_width)
        y = idx // int(self._input_width)
        if y >= int(self._input_height):
            return False
        return bool(self._valid_cycles[y, x])

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
        window = buf[:,-self._kernel_width:]
        window = window.astype(int, copy=False)
        result = np.zeros(self._channels, dtype=int)
        # Now take the dot product between the window, and the kernel
        for ch in range(self._channels):
            result[ch] = int((self.k[ch] * window).sum())
        return result

    def consume(self):
        """Called on each INPUT handshake.
        Returns:
          - None if this input position should NOT produce an output
          - int expected value if it SHOULD produce an output
        """
        assert_resolvable(self._data_i)
        inp = int(self._data_i.value.integer)

        # advance window on EVERY accepted input
        self._buf = self.update_window(self._buf, inp)

        idx = self._enqs
        x = idx % int(self._input_width)
        y = idx // int(self._input_width)
        self._enqs += 1

        if y >= int(self._input_height):
            return None

        if not self._valid_cycles[y, x]:
            return None

        # compute expected NOW, while _buf matches this accepted input position
        expected = self.apply_kernel(self._buf)
        return expected

    def produce(self, expected):
        assert_resolvable(self._data_o)

        # Per-channel width in bits (ConvOutWidth)
        w = int(self._dut.WidthOut.value) if hasattr(self._dut, "WidthOut") else int(self._dut.ConvOutWidth.value)
        # ^ if WidthOut param isn't on dut, use the same width you used for the RTL data_o element

        packed = int(self._data_o.value.integer)  # whole [OutChannels][WidthOut] blob as an int

        for ch in range(self._channels):
            raw = (packed >> (ch * w)) & ((1 << w) - 1)   # assumes ch=0 is LSB slice
            got = sign_extend(raw, w)
            exp = int(expected[ch])

            assert got == exp, (
                f"Mismatch at output #{self._deqs} ch{ch}: expected {exp}, got {got} "
                f"(raw=0x{raw:x})"
            )

class ReadyValidInterface():
    def __init__(self, clk, reset, ready, valid):
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
        self._width_p = int(dut.WidthIn.value)

    def generate(self):
        x_i = random.randint(0, (1 << self._width_p) - 1)
        return (x_i)

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

                success = True if ready_i.value == 1 and valid_o.value == 1 else False
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

            w = int(self._dut.WidthIn.value)

            if w == 1:
                # 1-bit data: allow 0/1
                data_i.value = din & 0x1
            else:
                # Keep MSB clear -> only non-negative signed values (0 .. 2^(w-1)-1)
                data_i.value = din & ((1 << (w - 1)) - 1)

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
    def __init__(self, dut, model):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i

        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.valid_i, dut.ready_o)
        self._rv_out = ReadyValidInterface(self._clk_i, self._rst_i,
                                           dut.valid_o, dut.ready_i)

        self._model = model
        self._events = queue.SimpleQueue()

        self._coro_run_in = None
        self._coro_run_out = None

    def start(self):
        """Start model"""
        if self._coro_run_in is not None or self._coro_run_out is not None:
            raise RuntimeError("Model already started")
        self._coro_run_in  = cocotb.start_soon(self._run_input())
        self._coro_run_out = cocotb.start_soon(self._run_output())

    async def _run_input(self):
        while True:
            await self._rv_in.handshake(None)
            exp = self._model.consume()     # exp is None or int
            if exp is not None:
                self._events.put(exp)

    async def _run_output(self):
        while True:
            await self._rv_out.handshake(None)
            assert self._events.qsize() > 0, "Error! Module produced output without expected input"
            expected = self._events.get()
            self._model.produce(expected)

    def stop(self):
        """Stop model"""
        if self._coro_run_in is None and self._coro_run_out is None:
            raise RuntimeError("Model never started")
        if self._coro_run_in is not None:
            self._coro_run_in.kill()
            self._coro_run_in = None
        if self._coro_run_out is not None:
            self._coro_run_out.kill()
            self._coro_run_out = None
    

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
    """Drive pixels until the first VALID kernel position, then expect 1 output."""

    W = int(dut.LineWidthPx.value)
    K = int(dut.KernelWidth.value)
    C = int(dut.OutChannels.value)

    # Number of accepted inputs until first valid output position (x=K-1, y=K-1)
    N_first = (K - 1) * W + (K - 1) + 1

    # We expect exactly ONE output for this test (the first valid position)
    N_out = 1

    rate = 1

    kernel = [[[1]*K for _ in range(K)] for _ in range(C)]
    model = ConvLayerModel(dut, kernel, input_height=K)
    m = ModelRunner(dut, model)

    om = OutputModel(dut, RateGenerator(dut, 1), N_out)               # consume 1 output
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_first)  # produce N_first inputs

    dut.ready_i.value = 0
    dut.valid_i.value = 0
    WeightWidth = int(dut.WeightWidth.value)
    weights_by_ch = [[1]*(K*K) for _ in range(C)]
    dut.weights_i.value = pack_weights_channels(weights_by_ch, WeightWidth, K*K)

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait until that single output is observed; timeout in ns but generous
    # If your clk is 1ns, N_first cycles is ~N_first ns; add cushion
    tmo_ns = 4 * N_first + 50

    timed_out = False
    try:
        await om.wait(tmo_ns)
    except SimTimeoutError:
        timed_out = True

    assert not timed_out, (
        f"Timed out waiting for first valid output. "
        f"W={W}, K={K}, expected after ~{N_first} accepted inputs."
    )

    dut.valid_i.value = 0
    dut.ready_i.value = 0

@cocotb.test
async def out_fuzz_test(dut):
    """Consumer fuzzed (ready_i), producer full-rate.

    DUT only outputs for valid kernel positions (x>=K-1 && y>=K-1), so:
      - l_out counts VALID outputs to observe
      - N_in is inputs to drive (warmup + l_out valid outputs)
    """

    W = int(dut.LineWidthPx.value)
    K = int(dut.KernelWidth.value)
    C = int(dut.OutChannels.value)
    S = int(dut.Stride.value)
    O = int(dut.StrideOrigin.value) if hasattr(dut, "StrideOrigin") else 0  # optional

    # Observe 4 rows of VALID outputs
    H_out = 4
    W_out = ((W - K) // S) + 1
    l_out = W_out * H_out

    y_last = (K - 1) + (H_out - 1) * S
    N_in = (y_last * W + (W - 1)) + 1
    input_height = y_last + 1

    # Consumer ready probability
    rate = 0.5

    first_out_wait_ns = 2 * (K - 1) * W + 2 * (K - 1) + 200
    timeout_ns = H_out * N_in + 500

    kernel = [[[1] * K for _ in range(K)] for _ in range(C)]
    weights_by_ch = [[1]*(K*K) for _ in range(C)]
    WeightWidth = int(dut.WeightWidth.value)
    dut.weights_i.value = pack_weights_channels(weights_by_ch, WeightWidth, K*K)

    model = ConvLayerModel(dut, kernel, input_height=input_height)
    m = ModelRunner(dut, model)

    # Consumer fuzzed; producer always drives valid
    om = OutputModel(dut, CountingGenerator(dut, rate), l_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, 1), N_in)

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # First output wait: producer is full rate, but DUT may stall due to consumer backpressure.
    # Give a bound proportional to N_first and 1/rate.
    # Wait until valid_o ever asserts (not necessarily handshake)
    try:
        await with_timeout(RisingEdge(dut.valid_o), first_out_wait_ns, 'ns')
    except SimTimeoutError:
        assert 0, (
            f"Timed out waiting for valid_o high. "
            f"W={W}, K={K}, S={S}, H_out={H_out}, W_out={W_out}, N_in={N_in}, waited={first_out_wait_ns} ns."
        )

    # Now wait for exactly l_out output handshakes
    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, (
            f"Timed out. Expected {l_out} output handshakes "
            f"(W_out={W_out}, H_out={H_out}). Got {om.nproduced()} in {timeout_ns} ns. "
            f"N_in={N_in}, input_height={input_height}."
        )


@cocotb.test
async def in_fuzz_test(dut):
    """Producer fuzzed (valid_i), consumer always-ready.
    Verify 4 output rows worth of VALID outputs under stride S.
    """

    W = int(dut.LineWidthPx.value)
    K = int(dut.KernelWidth.value)
    C = int(dut.OutChannels.value)
    S = int(dut.Stride.value)

    # ---- how many outputs to observe ----
    H_out = 4
    W_out = ((W - K) // S) + 1
    l_out = W_out * H_out

    # ---- how many inputs to drive (must cover H_out output rows) ----
    y_last = (K - 1) + (H_out - 1) * S
    N_in = (y_last * W + (W - 1)) + 1
    input_height = y_last + 1

    # Producer valid probability
    rate = 0.5

    kernel = [[[1] * K for _ in range(K)] for _ in range(C)]
    model = ConvLayerModel(dut, kernel, input_height=input_height)
    m = ModelRunner(dut, model)

    # Consumer always ready; producer fuzzed
    om = OutputModel(dut, RateGenerator(dut, 1), l_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_in)

    dut.ready_i.value = 0
    dut.valid_i.value = 0
    weights_by_ch = [[1] * (K * K) for _ in range(C)]
    WeightWidth = int(dut.WeightWidth.value)
    dut.weights_i.value = pack_weights_channels(weights_by_ch, WeightWidth, K * K)

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Need enough ACCEPTED inputs to reach the first possible output.
    # With producer rate 'rate', accepted inputs arrive ~rate per cycle when ready_o stays high.
    # Use N_first for the first kernel-valid position, and scale by 1/rate.
    N_first = (K - 1) * W + (K - 1) + 1
    first_out_wait_ns = int(4 * (N_first / rate)) + 200
    try:
        await with_timeout(RisingEdge(dut.valid_o), first_out_wait_ns, 'ns')
    except SimTimeoutError:
        assert 0, (
            f"Timed out waiting for valid_o high. "
            f"W={W}, K={K}, S={S}, N_first={N_first}, prod_rate={rate}, waited={first_out_wait_ns} ns."
        )

    # Total timeout: need to ACCEPT N_in inputs at rate~rate (consumer always-ready)
    timeout_ns = int(6 * (N_in / rate)) + 500
    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, (
            f"Timed out. Expected {l_out} output handshakes (W_out={W_out}, H_out={H_out}). "
            f"Only got {om.nproduced()} in {timeout_ns} ns. "
            f"N_in={N_in}, prod_rate={rate}, input_height={input_height}."
        )


@cocotb.test
async def inout_fuzz_test(dut):
    """Both producer and consumer are fuzzed.
    Verify 4 output rows worth of VALID outputs under stride S.
    """

    W = int(dut.LineWidthPx.value)
    K = int(dut.KernelWidth.value)
    C = int(dut.OutChannels.value)
    S = int(dut.Stride.value)

    # ---- outputs to observe ----
    H_out = 4
    W_out = ((W - K) // S) + 1
    l_out = W_out * H_out

    # ---- inputs to drive (cover H_out output rows) ----
    y_last = (K - 1) + (H_out - 1) * S
    N_in = (y_last * W + (W - 1)) + 1
    input_height = y_last + 1

    # Both sides fuzzed
    rate = 0.5  # probability valid_i=1 and ready_i=1

    kernel = [[[1] * K for _ in range(K)] for _ in range(C)]
    model = ConvLayerModel(dut, kernel, input_height=input_height)
    m = ModelRunner(dut, model)

    om = OutputModel(dut, RateGenerator(dut, rate), l_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_in)

    dut.ready_i.value = 0
    dut.valid_i.value = 0
    weights_by_ch = [[1] * (K * K) for _ in range(C)]
    WeightWidth = int(dut.WeightWidth.value)
    dut.weights_i.value = pack_weights_channels(weights_by_ch, WeightWidth, K * K)

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # First output wait: need ~N_first accepted inputs.
    # Accepted inputs occur when valid_i && ready_o. With producer rate=rate and moderate backpressure,
    # expect O(N_first/rate) cycles; be generous.
    N_first = (K - 1) * W + (K - 1) + 1
    first_out_wait_ns = int(6 * (N_first / rate)) + 300
    try:
        await with_timeout(RisingEdge(dut.valid_o), first_out_wait_ns, 'ns')
    except SimTimeoutError:
        assert 0, (
            f"Timed out waiting for valid_o high. "
            f"W={W}, K={K}, S={S}, N_first={N_first}, rate={rate}, waited={first_out_wait_ns} ns."
        )

    # Total timeout: need N_in accepted inputs (~N_in/rate cycles) and l_out output handshakes.
    # Output handshakes require ready_i=1 (prob=rate) so effective service rate is ~rate as well.
    timeout_ns = int(8 * (N_in / rate) + 8 * (l_out / rate)) + 800

    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, (
            f"Timed out. Expected {l_out} output handshakes (W_out={W_out}, H_out={H_out}). "
            f"Only got {om.nproduced()} in {timeout_ns} ns. "
            f"N_in={N_in}, rate={rate}, input_height={input_height}."
        )

        
@cocotb.test
async def full_bw_test(dut):
    """Transmit data at 100% line rate; verify 4 output rows worth of valid outputs."""

    W = int(dut.LineWidthPx.value)
    K = int(dut.KernelWidth.value)
    C = int(dut.OutChannels.value)
    S = int(dut.Stride.value)
    O = int(dut.StrideOrigin.value) if hasattr(dut, "StrideOrigin") else 0  # optional

    # ---- how many outputs to observe ----
    H_out = 4
    W_out = ((W - K) // S) + 1                 # valid conv/pool output width
    l_out = W_out * H_out

    # ---- how many inputs to feed to guarantee H_out output rows exist ----
    # last output row index is (H_out-1), which maps to input-origin row:
    # y_last = (K-1) + (H_out-1)*S
    y_last = (K - 1) + (H_out - 1) * S
    N_in = (y_last * W + (W - 1)) + 1          # stream through end of that row
    input_height = y_last + 1                  # number of rows actually streamed

    rate = 1

    # timeouts (ns): assume 1ns clock; add cushion
    first_out_wait_ns = 2 * (K - 1) * W + 2 * (K - 1) + 200
    timeout_ns = 4 * N_in + 500

    # weights
    kernel = [[[1] * K for _ in range(K)] for _ in range(C)]
    weights_by_ch = [[1] * (K * K) for _ in range(C)]
    WeightWidth = int(dut.WeightWidth.value)
    dut.weights_i.value = pack_weights_channels(weights_by_ch, WeightWidth, K * K)

    # models
    model = ConvLayerModel(dut, kernel, input_height=input_height)
    m = ModelRunner(dut, model)

    om = OutputModel(dut, RateGenerator(dut, 1), l_out)                     # consumer always ready (via generator)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_in)

    # init signals
    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait until valid_o ever asserts (not necessarily handshake)
    try:
        await with_timeout(RisingEdge(dut.valid_o), first_out_wait_ns, 'ns')
    except SimTimeoutError:
        assert 0, (
            f"Timed out waiting for valid_o high. "
            f"W={W}, K={K}, S={S}, H_out={H_out}, W_out={W_out}, N_in={N_in}, waited={first_out_wait_ns} ns."
        )

    # Now wait for exactly l_out output handshakes
    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, (
            f"Timed out. Expected {l_out} output handshakes "
            f"(W_out={W_out}, H_out={H_out}). Got {om.nproduced()} in {timeout_ns} ns. "
            f"N_in={N_in}, input_height={input_height}."
        )


@cocotb.test
async def full_bw_Gxy_test(dut):
    """Transmit data at 100% line rate with Gx/(Gy) kernel; verify 4 output rows worth of valid outputs."""

    W = int(dut.LineWidthPx.value)
    K = int(dut.KernelWidth.value)
    C = int(dut.OutChannels.value)
    S = int(dut.Stride.value)
    rate = 1

    # ---- outputs to observe ----
    H_out = 4
    W_out = ((W - K) // S) + 1
    l_out = W_out * H_out

    # ---- inputs to drive (cover H_out output rows under stride S) ----
    y_last = (K - 1) + (H_out - 1) * S
    N_in = (y_last * W + (W - 1)) + 1
    input_height = y_last + 1

    # Kernel weights
    if K == 3:
        gx_k = [[-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]]
        gy_k = [[-1, -1, -1],
                [ 0,  0,  0],
                [ 1,  1,  1]]
    elif K == 5:
        gx_k = [[-1, -1, 0, 1, 1],
                [-1, -1, 0, 1, 1],
                [-1, -1, 0, 1, 1],
                [-1, -1, 0, 1, 1],
                [-1, -1, 0, 1, 1]]
        gy_k = [[-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [ 0,  0,  0,  0,  0],
                [ 1,  1,  1,  1,  1],
                [ 1,  1,  1,  1,  1]]
    else:
        assert 0, f"Unsupported kernel width {K}"

    assert C in (1, 2), f"Expected OutChannels 1 or 2, got {C}"
    kernel_by_ch = [gx_k] if C == 1 else [gx_k, gy_k]
    weights_by_ch_flat = [flatten_kernel(k) for k in kernel_by_ch]

    # Model + runner
    model = ConvLayerModel(dut, kernel_by_ch, input_height=input_height)
    m = ModelRunner(dut, model)

    # Always-ready consumer; full-rate producer
    om = OutputModel(dut, RateGenerator(dut, 1), l_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_in)

    # Init
    dut.ready_i.value = 0
    dut.valid_i.value = 0
    WeightWidth = int(dut.WeightWidth.value)
    dut.weights_i.value = pack_weights_channels(weights_by_ch_flat, WeightWidth, K * K)

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # First possible kernel-valid position (still useful for a "did anything ever start?" check)
    N_first = (K - 1) * W + (K - 1) + 1
    first_out_wait_ns = int(2 * N_first) + 200
    try:
        await with_timeout(RisingEdge(dut.valid_o), first_out_wait_ns, 'ns')
    except SimTimeoutError:
        assert 0, (
            f"Timed out waiting for valid_o to go high. "
            f"W={W}, K={K}, S={S}, N_first={N_first}, waited={first_out_wait_ns} ns. "
            f"(Expecting {l_out} outputs = {H_out} rows × W_out={W_out})"
        )

    # Full-bandwidth timeout scaled to N_in (and a cushion)
    timeout_ns = int(4 * N_in) + 500
    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, (
            f"Timed out. Expected {l_out} output handshakes (W_out={W_out}, H_out={H_out}). "
            f"Only got {om.nproduced()} in {timeout_ns} ns. "
            f"N_in={N_in}, input_height={input_height}, W={W}, K={K}, S={S}."
        )