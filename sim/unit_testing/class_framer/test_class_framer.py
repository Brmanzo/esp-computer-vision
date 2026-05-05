# test_class_framer.py
import os
from   pathlib import Path
import pytest

from util.utilities import runner, lint, assert_resolvable, clock_start_sequence, \
                           sim_verbose, reset_sequence, load_tests_from_csv, auto_unpack
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from functional_models.class_framer import ClassFramerModel
from util.gen_inputs import gen_random_unsigned
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import Decimal, Timer, RisingEdge, FallingEdge
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

# For class_framer, we only really care about BusBits
TEST_CASES = [{"BusBits": 8}]

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)
def test_each(test_name, simulator, BusBits):
    if isinstance(BusBits, (tuple, list)):
        BusBits = BusBits[0]
    parameters = {"BusBits": int(BusBits)}
    runner(simulator, timescale, tbpath, parameters, testname=test_name, pymodule="test_class_framer")

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""
    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)

class ClassDataGenerator():
    def __init__(self, dut):
        self._width = int(dut.BusBits.value)

    def generate(self):
        val = gen_random_unsigned(self._width, rng=random)
        return val, val

@cocotb.test
async def single_test(dut):
    """Test to transmit a single classification result (3 bytes total output)."""
    
    n_classifications = 1
    # 1 input per classification, 3 outputs (Class + Tail0 + Tail1)
    l_in = n_classifications 
    l_out = n_classifications * 3 + 1 # +1 for the initial Wakeup byte
    rate = 1.0

    m = ModelRunner(dut, ClassFramerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, rate), l_out)
    im = InputModel(dut, ClassDataGenerator(dut), RateGenerator(dut, rate), l_in, data_pins=dut.class_i)

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    # Pre-inject the expected Wakeup byte so the ModelRunner validates it correctly
    m._events.put(int(dut.WakeupCmd.value))

    m.start()
    om.start()
    im.start()

    timeout_ns = 1000
    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, f"Timed out waiting for {l_out} bytes"

async def rate_tests(dut, in_rate, out_rate):
    """Stress test with different input/output speeds"""
    n_classifications = 10
    l_in = n_classifications
    l_out = n_classifications * 3 + 1 

    m = ModelRunner(dut, ClassFramerModel(dut))
    om = OutputModel(dut, RateGenerator(dut, out_rate), l_out)
    im = InputModel(dut, ClassDataGenerator(dut), RateGenerator(dut, in_rate), l_in, data_pins=dut.class_i)

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m._events.put(int(dut.WakeupCmd.value))

    m.start()
    om.start()
    im.start()

    slow = min(in_rate, out_rate)
    timeout_ns = int((l_out * 100) / slow) # Generous timeout
    
    await om.wait(timeout_ns)

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
