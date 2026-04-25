# test_multi_delay_ram.py
from   pathlib import Path
import pytest
import queue

from util.utilities import runner, lint, clock_start_sequence, reset_sequence, delay_cycles
from functional_models.models import MultiDelayBufferModel
from util.bitwise import pack_terms
from util.gen_inputs import gen_input_channels
from util.components import RateGenerator, InputModel
tbpath = Path(__file__).parent

import cocotb
from   cocotb.types import Logic
from   cocotb.triggers import RisingEdge, FallingEdge, with_timeout
from   cocotb.result import SimTimeoutError

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

def channels_per_ram(InBits: int, LineWidthPx: int, KernelWidth: int):
    if (LineWidthPx - 1) <= 256: TargetRamBits = 16
    else: TargetRamBits = 8
    return str(TargetRamBits // ((KernelWidth - 1) * InBits))

def buffer_count(InBits: int, LineWidthPx: int, KernelWidth: int, InChannels: int):
    ChannelsPerRam = int(channels_per_ram(InBits, LineWidthPx, KernelWidth))
    return str((InChannels + ChannelsPerRam - 1) // ChannelsPerRam)

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits, LineWidthPx, KernelWidth, ChannelsPerRam, InChannels, BufferCount", 
                         [("4", "258",  "3", channels_per_ram(4, 258, 3),  "1", buffer_count(4, 258, 3, 1))
                         ,("1", "258",  "9", channels_per_ram(1, 258, 9),  "1", buffer_count(1, 258, 9, 1))
                         ,("1", "258",  "3", channels_per_ram(1, 258, 3),  "4", buffer_count(1, 258, 3, 4))
                         ,("8", "257",  "3", channels_per_ram(8, 257, 3),  "1", buffer_count(8, 257, 3, 1))
                         ,("1", "257", "17", channels_per_ram(1, 257, 17), "1", buffer_count(1, 257, 17, 1))
                         ,("1", "257",  "3", channels_per_ram(1, 257, 3),  "8", buffer_count(1, 257, 3, 8))
                         ,("1", "260",  "3", channels_per_ram(1, 260, 3),  "32", buffer_count(1, 260, 3, 32))
                         ,("4", "258",  "3", channels_per_ram(4, 258, 3),  "2", buffer_count(4, 258, 3, 2))])
def test_each(test_name, simulator, InBits, LineWidthPx, KernelWidth, ChannelsPerRam, InChannels, BufferCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name)

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits, LineWidthPx, KernelWidth, ChannelsPerRam, InChannels, BufferCount", 
                         [("1", "8", "2", channels_per_ram(1, 8, 2), "1", buffer_count(1, 8, 2, 1))
                         ])
def test_all(simulator, InBits, LineWidthPx, KernelWidth, ChannelsPerRam, InChannels, BufferCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, LineWidthPx, KernelWidth, ChannelsPerRam, InChannels, BufferCount", 
                         [("1", "8", "2", channels_per_ram(1, 8, 2), "1", buffer_count(1, 8, 2, 1))
                         ])
def test_lint(simulator, InBits, LineWidthPx, KernelWidth, ChannelsPerRam, InChannels, BufferCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, LineWidthPx, KernelWidth, ChannelsPerRam, InChannels, BufferCount", 
                         [("1", "8", "2", channels_per_ram(1, 8, 2), "1", buffer_count(1, 8, 2, 1))
                         ])
def test_style(simulator, InBits, LineWidthPx, KernelWidth, ChannelsPerRam, InChannels, BufferCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

def unpack_data_o_top(InBits, KernelWidth, InChannels, packed_o):
    rows = KernelWidth - 1
    mask = (1 << InBits) - 1

    out = [[0] * rows for _ in range(InChannels)]

    for ch in range(InChannels):
        for r in range(rows):
            bitpos = (ch * rows + r) * InBits
            out[ch][r] = (packed_o >> bitpos) & mask

    return out

class MultiDelayRamModel():
    def __init__(self, dut):
        self._dut = dut
        self._data_i = dut.data_i
        self._data_o = dut.data_o

        self._q = queue.SimpleQueue()

        # Top Level connections
        self._BufferCount    = int(dut.BufferCount.value)
        self._ChannelsPerRam = int(dut.ChannelsPerRam.value)
        self._InBits         = int(dut.InBits.value)
        self._InChannels     = int(dut.InChannels.value)
        self._KernelWidth    = int(dut.KernelWidth.value)
        self._LineWidthPx    = int(dut.LineWidthPx.value)
        
        self._warmup = (self._LineWidthPx - 1) * (self._KernelWidth - 1) + 1
        
        # Individual RAM connections
        self._buffers = [
            MultiDelayBufferModel(
                BufferWidth   = self._InBits,
                Delay         = self._LineWidthPx - 1,
                BufferRows    = self._KernelWidth - 1,
                InputChannels = self._ChannelsPerRam
            )
            for _ in range(self._BufferCount)
        ]

    def step(self, data_i_words, in_fire=True):
        assert len(data_i_words) == self._InChannels
        
        full_out: list[list[int] | None] = [None] * self._InChannels
        any_valid = False

        for buf_idx, buf in enumerate(self._buffers):
            first_ch = buf_idx * self._ChannelsPerRam
            last_ch = first_ch + self._ChannelsPerRam

            chunk_in = list(map(int, data_i_words[first_ch:min(last_ch, self._InChannels)]))
            chunk_in += [0] * (self._ChannelsPerRam - len(chunk_in))

            chunk_out = buf.step(chunk_in, in_fire=True)

            if chunk_out is not None:
                any_valid = True
                for ch, out_word in enumerate(chunk_out):
                    idx = first_ch + ch
                    if idx < self._InChannels:
                        full_out[idx] = out_word

        return full_out if any_valid else None
    
class RandomDataGenerator:
    def __init__(self, dut):
        self._bw = int(dut.InBits.value)
        self._ic = int(dut.InChannels.value)
        self._first_high = False

    def generate(self):
        if not self._first_high:
            self._first_high = True
            raw = [-1] * self._ic 
        else:
            raw = gen_input_channels(self._bw, self._ic)

        mask = (1 << (self._bw - 1)) - 1 if self._bw > 1 else 0x1
        masked = [x & mask for x in raw]

        return pack_terms(masked, self._bw), masked

class OutputModel():
    def __init__(self, dut, model, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut
        self._model = model
        self._length = l

        self._coro = None
        self._nout = 0

    def start(self):
        if self._coro is not None:
            raise RuntimeError("Output Model already started")
        self._coro = cocotb.start_soon(self._run())

    def stop(self):
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
        self._nout = 0
        clk_i = self._clk_i
        rst_i = self._rst_i

        await FallingEdge(clk_i)

        if not (rst_i.value.is_resolvable and rst_i.value == 0):
            await FallingEdge(rst_i)

        while self._nout < self._length:
            await RisingEdge(clk_i)

            if not self._model._events.empty():
                expected_words = self._model._events.get()

                val = self._dut.data_o.value
                if not val.is_resolvable:
                    raise AssertionError(f"data_o contains X/Z: {val.binstr}")

                got_packed = int(val)
                got = unpack_data_o_top(
                    InBits=int(self._dut.InBits.value),
                    KernelWidth=int(self._dut.KernelWidth.value),
                    InChannels=int(self._dut.InChannels.value),
                    packed_o=got_packed
                )

                # print(f"OutputModel Check: got={got} expected={expected_words}")
                assert got == expected_words, f"Mismatch: got={got} expected={expected_words}"
                self._nout += 1

            await FallingEdge(clk_i)

        return self._nout
    
class ModelRunner():
    def unpack_data_i(self, packed_i: int):
        C = int(self._dut.InChannels.value)
        W = int(self._dut.InBits.value)
        mask = (1 << W) - 1

        words = [0] * C
        for ch in range(C):
            words[ch] = (packed_i >> (ch * W)) & mask
        return words

    def __init__(self, dut, model):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut
        self._model = model

        self._events = queue.SimpleQueue()
        self._coro_run_in = None
        self._coro_run_out = None

    def start(self):
        if self._coro_run_in is not None or self._coro_run_out is not None:
            raise RuntimeError("Model already started")
        self._coro_run_in  = cocotb.start_soon(self._run_input())
        self._coro_run_out = cocotb.start_soon(self._run_output())

    def stop(self):
        if self._coro_run_in is not None:
            self._coro_run_in.kill()
            self._coro_run_in = None
        if self._coro_run_out is not None:
            self._coro_run_out.kill()
            self._coro_run_out = None

    async def _run_input(self):
        while True:
            await RisingEdge(self._clk_i)

            if int(self._dut.rst_i.value) == 1:
                continue

            if int(self._dut.in_fire.value) == 1:
                packed_in = int(self._dut.data_i.value)
                words = self.unpack_data_i(packed_in)

                expected = self._model.step(words, in_fire=True)
                if expected is not None:
                    self._events.put(expected)

    async def _run_output(self):
        while True:
            await RisingEdge(self._clk_i)

            if int(self._dut.rst_i.value) == 1:
                continue

            if self._events.empty():
                continue

            expected_words = self._events.get()

            val = self._dut.data_o.value
            if not val.is_resolvable:
                raise AssertionError(f"data_o contains X/Z: {val.binstr}")

            got_packed = int(val)
            got = unpack_data_o_top(
                InBits=int(self._dut.InBits.value),
                KernelWidth=int(self._dut.KernelWidth.value),
                InChannels=int(self._dut.InChannels.value),
                packed_o=got_packed
            )

            print(f"OutputModel Check: got={got} expected={expected_words}")
            assert got == expected_words, f"Mismatch: got={got} expected={expected_words}"
            
async def flush_dut(dut, duration):
    """
    Drives 0s into the DUT to overwrite any old data in the RAM.
    Does not check output, effectively ignoring 'garbage' from previous tests.
    """
    dut.in_fire.value = 1
    dut.data_i.value = 0 
    
    for _ in range(duration):
        await RisingEdge(dut.clk_i)
        
    dut.in_fire.value = 0

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

    D = int(dut.LineWidthPx.value) - 1
    rows = int(dut.KernelWidth.value) - 1

    # Number of accepted inputs until first valid output position
    N_first = rows * D + 2

    # We expect exactly ONE output for this test
    N_out = 1

    rate = 1

    model = MultiDelayRamModel(dut)
    m = ModelRunner(dut, model)

    om = OutputModel(dut, m, N_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_first,
                    data_pins=dut.data_i, valid_pin=dut.in_fire, ready_pin=Logic(1))

    dut.in_fire.value = 0

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
        dut.in_fire.value = 0

        # Give one cycle to settle before next test
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)

async def rate_tests(dut, in_rate, out_rate):

    D = int(dut.LineWidthPx.value) - 1
    rows = int(dut.KernelWidth.value) - 1


    await clock_start_sequence(dut.clk_i)

    await reset_sequence(dut.clk_i, dut.rst_i, 10)


    flush_depth = rows * D + 5
    await flush_dut(dut, flush_depth)

    await reset_sequence(dut.clk_i, dut.rst_i, 10)

    model = MultiDelayRamModel(dut)
    m = ModelRunner(dut, model)

    # Number of accepted inputs until first valid output position (x=K-1, y=K-1)
    N_in = rows * D * 2 + 2 + model._warmup

    # We expect exactly ONE output for this test (the first valid position)
    N_out = N_in - model._warmup

    om = OutputModel(dut, m, N_out)             # consume N_out outputs
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, in_rate), N_in,
                    data_pins=dut.data_i, valid_pin=dut.in_fire, ready_pin=Logic(1))
    dut.in_fire.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    flush_depth = rows * D + 5
    await flush_dut(dut, flush_depth)
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
        dut.in_fire.value = 0

        # Give one cycle to settle before next test
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)

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