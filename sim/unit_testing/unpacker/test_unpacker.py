import os
from   pathlib import Path
import pytest

from util.utilities import runner, lint, assert_resolvable, clock_start_sequence, \
                           reset_sequence, load_tests_from_csv, auto_unpack, \
                           sim_verbose
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from util.gen_inputs import gen_random_unsigned
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import FallingEdge
from   cocotb.result import SimTimeoutError
   
import random
random.seed(42)

timescale = "1ps/1ps"

tests = ['reset_test'
        ,'single_test'
        ,'inout_fuzz_test'
        ,'in_fuzz_test'
        ,'out_fuzz_test'
        ,'full_bw_test']

TEST_CASES = load_tests_from_csv(os.path.join(tbpath, "test_cases.csv"))
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)
def test_each(test_name, simulator, UnpackedWidth, PackedNum):
    parameters = dict(locals())
    parameters.pop( 'test_name', None)
    parameters.pop('simulator', None)
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_unpacker")

@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)
def test_all(simulator, UnpackedWidth, PackedNum):
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, pymodule="test_unpacker")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator):
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, pymodule="test_unpacker")

@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator):
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"], pymodule="test_unpacker")

class UnpackerModel():
    def __init__(self, dut):
        self._dut = dut
        self._unpacked_o = dut.unpacked_o
        self._packed_i = dut.packed_i
        self._UnpackedWidth = dut.UnpackedWidth.value
        self._PackedNum = dut.PackedNum.value
        self._PackedWidth = dut.PackedWidth.value

        self._mask = (1 << self._UnpackedWidth) - 1
        self._deqs = 0
    
    def consume(self):
        assert_resolvable(self._packed_i)
        packed_val = int(self._packed_i.value) & ((1 << (self._PackedWidth)) - 1)
        expected_outputs = []
        for step in range(self._PackedNum):
            val = (packed_val >> (self._UnpackedWidth * step)) & self._mask
            expected_outputs.append(val)

        return expected_outputs

    def produce(self, expected):
        assert_resolvable(self._unpacked_o)
        got = int(self._unpacked_o.value) & self._mask

        self._deqs += 1
        if sim_verbose():
            print(f'Output #{self._deqs}: Got unpacked: {got}, Expected: {expected}')

        assert got == expected, (
            f"Mismatch on output #{self._deqs}: expected {expected}, got {got}"
        )

class RandomDataGenerator():
    def __init__(self, dut):
        self._dut = dut
        self._width_p = dut.PackedWidth.value

    def generate(self):
        val = gen_random_unsigned(self._width_p, random)
        return val, val

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

@cocotb.test
async def single_test(dut):
    """Test to transmit a single element."""

    l = 1
    eg = RandomDataGenerator(dut)
    rate = 1
   
    PackedNum = int(dut.PackedNum.value)
   
    model = UnpackerModel(dut)
    m = ModelRunner(dut, model)
    
    # 1 item in = PackedNum items out
    om = OutputModel(dut, RateGenerator(dut, 1), l * PackedNum) 
    
    im = InputModel(dut, eg, RateGenerator(dut, rate), l, data_pins=dut.packed_i)

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
    
    # The fast start-up of the generic InputModel caused this edge to be missed, deadlocking the test.

    timeout = False
    try:
        await om.wait(200) 
    except SimTimeoutError:
        timeout = True
    assert not timeout, "Error! Circuit took too long to yield the expected output vectors."

    dut.valid_i.value = 0
    dut.ready_i.value = 0

async def rate_tests(dut, in_rate, out_rate):
    """Input random data elements at 100% line rate"""

    eg = RandomDataGenerator(dut)
    
    PackedNum = int(dut.PackedNum.value)
    l_in = 10
    l_out = l_in * PackedNum 
    
    timeout_cycles = int((l_in / in_rate) + (l_out / out_rate)) + 100
    timeout_ns = timeout_cycles * 10

    m = ModelRunner(dut, UnpackerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, out_rate), l_out)
    
    im = InputModel(dut, eg, RateGenerator(dut, in_rate), l_in, data_pins=dut.packed_i)

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

    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, f"Test timed out. Could not transmit {l_out} elements in {timeout_ns} ns, with output rate {out_rate}"

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