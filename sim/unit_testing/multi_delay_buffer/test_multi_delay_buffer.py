# test_multi_delay_buffer.py
from   pathlib import Path
import pytest

from util.utilities import runner, lint, clock_start_sequence, reset_sequence
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from util.gen_inputs import gen_input_channels
from util.bitwise import pack_terms
from functional_models.models import MultiDelayBufferModel
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import RisingEdge, FallingEdge
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

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("BufferWidth, Delay, BufferRows, InputChannels", 
                         [("1", "8", "1", "1"), ("1", "8", "2", "4"), ("2", "16", "4", "10")])
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
    
class RandomDataGenerator():
    def __init__(self, dut):
        self._bw = int(dut.BufferWidth.value)
        self._ic = int(dut.InputChannels.value)
        self._first_high = False

    def generate(self):
        if not self._first_high:
            self._first_high = True
            raw = [-1] * self._ic 
        else:
            raw = gen_input_channels(self._bw, self._ic)

        if self._bw == 1:
            masked = [x & 0x1 for x in raw]
        else:
            masked = [x & ((1 << (self._bw - 1)) - 1) for x in raw]

        return pack_terms(masked, self._bw), masked

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

async def rate_tests(dut, in_rate, out_rate):

    D = int(dut.Delay.value)
    rows = int(dut.BufferRows.value)

    await clock_start_sequence(dut.clk_i)

    await reset_sequence(dut.clk_i, dut.rst_i, 10)


    flush_depth = rows * D + 5
    await flush_dut(dut, flush_depth)

    await reset_sequence(dut.clk_i, dut.rst_i, 10)

    model = MultiDelayBufferModel(dut)
    m = ModelRunner(dut, model)

    # Number of accepted inputs until first valid output position (x=K-1, y=K-1)
    N_in = (dut.BufferRows.value)*D*2 + 2 + model._warmup

    # We expect exactly ONE output for this test (the first valid position)
    N_out = N_in - model._warmup

    om = OutputModel(dut, RateGenerator(dut, out_rate), N_out)               # consume N_out outputs
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, in_rate), N_in)  # produce N_in inputs

    dut.ready_i.value = 0
    dut.valid_i.value = 0

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