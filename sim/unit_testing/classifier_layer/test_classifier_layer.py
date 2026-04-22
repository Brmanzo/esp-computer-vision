# test_classifier_layer.py
import git
import os
import sys
import numpy as np
import queue
from itertools import product
from decimal import Decimal

# I don't like this, but it's convenient.
_REPO_ROOT = git.Repo(search_parent_directories=True).working_tree_dir
assert _REPO_ROOT is not None, "REPO_ROOT path must not be None"
assert (os.path.exists(_REPO_ROOT)), "REPO_ROOT path must exist"
sys.path.append(os.path.join(_REPO_ROOT, "util"))
from utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence, delay_cycles, ReadyValidInterface
tbpath = os.path.dirname(os.path.realpath(__file__))

import pytest

import cocotb

from cocotb.utils import get_sim_time
from cocotb.triggers import Timer, RisingEdge, FallingEdge, with_timeout
from cocotb.result import SimTimeoutError
   
import random
random.seed(42)

timescale = "1ps/1ps"

def sign_extend(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    sign_bit = 1 << (width - 1)
    return (value ^ sign_bit) - sign_bit

def pack(inputs, term_bits):
    packed = 0
    mask = (1 << term_bits) - 1
    for i, x in enumerate(inputs):
        packed |= (x & mask) << (i * term_bits)
    return packed

def unpack(packed, term_bits, input_count):
    unpacked = []
    mask = (1 << term_bits) - 1
    for i in range(input_count):
        raw = (packed >> (i * term_bits)) & mask
        unpacked.append(sign_extend(raw, term_bits))
    return unpacked

def trunc_signed(value: int, width: int) -> int:
    return sign_extend(value, width)

tests = ['reset_test'
        ,'single_test'
        ,'inout_fuzz_test'
        ,'in_fuzz_test'
        ,'out_fuzz_test'
        ,'full_bw_test']

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("TermBits, ClassCount, TermCount", [
    (8, 9, 32),
    (16, 17, 64),
    (32, 33, 64)
])
def test_each(test_name, simulator, TermBits, ClassCount, TermCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_classifier_layer")

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("TermBits, ClassCount, TermCount", [
    (8, 9, 32)
])
def test_all(simulator, TermBits, ClassCount, TermCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, pymodule="test_classifier_layer")

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("TermBits, ClassCount, TermCount", [
    (8, 9, 32)
])
def test_lint(simulator, TermBits, ClassCount, TermCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, pymodule="test_classifier_layer")

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("TermBits, ClassCount, TermCount", [
    (8, 9, 32)
])
def test_style(simulator, TermBits, ClassCount, TermCount):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"], pymodule="test_classifier_layer")

class ClassifierLayerModel:
    def __init__(self, dut):
        self._dut = dut
        self._data_i = dut.data_i
        self._class_o = dut.class_o

        self._term_bits = int(dut.TermBits.value)
        self._class_count = int(dut.ClassCount.value)
        self._term_count = int(dut.TermCount.value)

        self._term_counter = 0
        self._current_max = None
        self._expected = queue.SimpleQueue()

    def consume(self):
        assert_resolvable(self._data_i)

        packed_in = int(self._data_i.value.integer)
        x = unpack(packed_in, self._term_bits, self._class_count)

        if self._term_counter == 0:
            self._current_max = x[:]
        else:
            for ch in range(self._class_count):
                self._current_max[ch] = max(self._current_max[ch], x[ch])

        self._term_counter += 1

        if self._term_counter == self._term_count:
            max_val = max(self._current_max)
            class_id = self._current_max.index(max_val)  # lowest index wins ties
            self._expected.put((class_id, self._current_max[:]))
            self._term_counter = 0
            self._current_max = None

    def produce(self):
        assert_resolvable(self._class_o)

        expected_id, max_vector = self._expected.get()
        got_id = int(self._class_o.value.integer)

        print(
            f"Produced class {got_id}, expected {expected_id}, "
            f"max_vector={max_vector} at time {get_sim_time(units='ns')}ns"
        )

        assert got_id == expected_id, (
            f"Class mismatch. Expected {expected_id}, got {got_id}"
        )

class RandomDataGenerator:
    def __init__(self, dut):
        self._dut = dut

    def generate(self):
        term_bits = int(self._dut.TermBits.value)
        class_count = int(self._dut.ClassCount.value)

        lo = -(1 << (term_bits - 1))
        hi =  (1 << (term_bits - 1)) - 1

        vals = [random.randint(lo, hi) for _ in range(class_count)]
        return pack(vals, term_bits)

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
            success = 0
            valid_i.value = produce
            data = self._data.generate()

            data_i.value = data if produce else 0

            # Wait until ready
            while(produce and not success):
                await RisingEdge(clk_i)
                assert_resolvable(ready_o)

                success = True if (ready_o.value == 1) else False
                if (success):
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
        self._coro_run_input = None
        self._coro_run_output = None

    def start(self):
        if self._coro_run_input is not None:
            raise RuntimeError("Model already started")
        self._coro_run_input = cocotb.start_soon(self._run_input(self._model))
        self._coro_run_output = cocotb.start_soon(self._run_output(self._model))

    async def _run_input(self, model):
        while True:
            await self._rv_in.handshake(None)
            self._model.consume()

    async def _run_output(self, model):
        while True:
            await self._rv_out.handshake(None)
            self._model.produce()

    def stop(self):
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
async def single_test(dut):
    T = int(dut.TermCount.value)

    model = ClassifierLayerModel(dut)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, 1), 1)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, 1), T)

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    timeout_ns = 200
    await om.wait(timeout_ns)

async def rate_tests(dut, in_rate, out_rate):
    """Input random data elements at 100% line rate"""

    eg = RandomDataGenerator(dut)
    T = int(dut.TermCount.value)
    groups = 10
    l_in = groups * T
    l_out = groups

    m = ModelRunner(dut, ClassifierLayerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, out_rate), l_out)
    im = InputModel(dut, eg, RateGenerator(dut, in_rate), l_in)

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

    slow = min(in_rate, out_rate)
    slow = max(slow, 0.05) 
    timeout_ns        = int((l_in + 500) / slow)

    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, (
            f"Test timed out. Expected {l_out} outputs from {l_in} inputs "
            f"with ClassCount={T}, in_rate={in_rate}, out_rate={out_rate}"
    )

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
