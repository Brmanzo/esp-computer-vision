import os
from   pathlib import Path
import pytest

from util.utilities  import runner, lint, clock_start_sequence, reset_sequence
tbpath = Path(__file__).parent

import cocotb
from   cocotb.utils import get_sim_time
from   cocotb.triggers import RisingEdge, FallingEdge, with_timeout
from   cocotb.types import LogicArray, Range
   
import random
random.seed(42)

timescale = "1ps/1ps"

timescale = "1ps/1ps"
tests = ['reset_test',
         'single_cycle_test',
         'limit_test_up',
         'limit_test_down',
         'fuzz_test_short',
         'fuzz_test_medium',
         'fuzz_test_long'
         ]

@pytest.mark.parametrize("ResetVal,MaxVal", [(1, 3), (11, 67), (30, 31), (0, 11)])
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
def test_each(test_name, simulator, ResetVal, MaxVal):
    # This line must be first
    parameters = dict(locals())
    parameters.pop('test_name', None)
    parameters.pop('simulator', None)
    runner(simulator, timescale, tbpath, parameters, testname=test_name)

@pytest.mark.parametrize("ResetVal,MaxVal", [(1, 3), (11, 67), (30, 31), (0, 11)])
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
def test_all(simulator, ResetVal, MaxVal):
    # This line must be first
    parameters = dict(locals())
    parameters.pop('simulator', None)
    runner(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("ResetVal, MaxVal", [(11, 67)])
@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator, ResetVal, MaxVal):
    # This line must be first
    parameters = dict(locals())
    parameters.pop('simulator', None)
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("ResetVal, MaxVal", [(11, 67)])
@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator, ResetVal, MaxVal):
    # This line must be first
    parameters = dict(locals())
    parameters.pop('simulator', None)
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

class CounterModel():
    def __init__(self, dut):

        self._CountBits   = dut.CountBits.value
        self._ResetVal    = dut.ResetVal.value
        self._MaxVal      = dut.MaxVal.value
        self._EnableDown  = dut.EnableDown.value
        self._clk_i       = dut.clk_i
        self._rst_i     = dut.rst_i
        self._up_i        = dut.up_i
        self._down_i      = dut.down_i
        self._count_o     = dut.count_o
        self._coro_run    = None
        self._count       = 0

    def start(self):
        """Start model"""
        if self._coro_run is not None:
            raise RuntimeError("Model already started")
        self._coro_run = cocotb.start_soon(self._run())

    async def _run(self):
        while True:
            await RisingEdge(self._clk_i)
            if(not(self._rst_i.value.is_resolvable)):
                pass
            elif(self._rst_i.value == 1):
                self._count = self._ResetVal
            elif(not(self._up_i.value.is_resolvable and self._down_i.value.is_resolvable)):
                pass
            elif(self._up_i.value and self._down_i.value):
                pass # Do nothing
            elif(self._up_i.value and not self._down_i.value and (self._count != self._MaxVal)):
                self._count += 1
            elif(self._up_i.value and not self._down_i.value):
                self._count = 0
            elif(not self._up_i.value and self._down_i.value and (self._count != 0)):
                self._count -= 1
            elif(not self._up_i.value and self._down_i.value):
                self._count = self._MaxVal
      
    def stop(self) -> None:
        """Stop monitor"""
        if self._coro_run is None:
            raise RuntimeError("Monitor never started")
        self._coro_run.cancel()
        self._coro_run = None
    
@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    up_i = dut.up_i
    down_i = dut.down_i
    count_o = dut.count_o

    up_i.value = 0
    down_i.value = 0

    await clock_start_sequence(clk_i)
    model = CounterModel(dut)
    model.start()
    await reset_sequence(clk_i, rst_i, 10)

    # Set the initial inputs
    up_i.value = 0
    down_i.value = 0

    await FallingEdge(dut.clk_i)

    assert count_o.value.is_resolvable, f"Unresolvable value (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."
    assert count_o.value == model._count , f"Incorrect Result: count_o != {model._count}. Got: {count_o.value} at Time {get_sim_time(units='ns')}ns."

@cocotb.test
async def single_cycle_test(dut):
    """Test for single cycle up/down"""
    await clock_start_sequence(dut.clk_i)
    model = CounterModel(dut)
    model.start()
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    
    # Increment once
    await FallingEdge(dut.clk_i)
    dut.up_i.value = 1
    dut.down_i.value = 0
    await FallingEdge(dut.clk_i)
    assert dut.count_o.value == model._count

async def wait_for(dut, value):
    while(dut.count_o.value.is_resolvable and dut.count_o.value != value):
        await FallingEdge(dut.clk_i)

async def limit_test(dut, up, down):
    """Test for MaxVal/0 limits."""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    up_i = dut.up_i
    down_i = dut.down_i
    count_o = dut.count_o

    up_i.value = LogicArray('x', Range(0, 'downto', 0))
    down_i.value = LogicArray('x', Range(0, 'downto', 0))

    await clock_start_sequence(clk_i)
    model = CounterModel(dut)
    model.start()
    await reset_sequence(clk_i, rst_i, 10)

    # Set the initial inputs
    up_i.value = 0
    down_i.value = 0
    
    await FallingEdge(dut.clk_i)
    up_i.value = up
    down_i.value = down

    value = None
    timeout = 0

    if(up):
        value = int(dut.MaxVal.value)
        timeout = int(dut.MaxVal.value) - int(dut.ResetVal.value) + 1
    elif(down):
        value = 0
        timeout = 2*int(dut.ResetVal.value) + 1
    
    await with_timeout(wait_for(dut, value=value), timeout * 100, 'ns')

    # Increment once more.
    await FallingEdge(dut.clk_i)

    assert count_o.value.is_resolvable, f"Unresolvable value (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."
    assert count_o.value == model._count_o , f"Incorrect Result: count_o != {model._count_o}. Got: {count_o.value} at Time {get_sim_time(units='ns')}ns."

@cocotb.test
async def limit_test_up(dut):
    await limit_test(dut, 1, 0)

@cocotb.test
async def limit_test_down(dut):
    await limit_test(dut, 0, 1)

async def fuzz_test(dut, l):
    """Test for Random Input"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    up_i = dut.up_i
    down_i = dut.down_i
    count_o = dut.count_o

    up_i.value = LogicArray('x', Range(0, 'downto', 0))
    down_i.value = LogicArray('x', Range(0, 'downto', 0))

    await clock_start_sequence(clk_i)
    model = CounterModel(dut)
    model.start()
    await reset_sequence(clk_i, rst_i, 10)

    # Set the initial inputs
    up_i.value = 0
    down_i.value = 0

    await FallingEdge(dut.clk_i)

    seq = [random.randint(0, 4) for i in range(l)]
    for i in seq:
        await FallingEdge(dut.clk_i)
        up_i.value = (i == 1 or i == 3)
        down_i.value = (i == 2 or i == 3)
        assert count_o.value.is_resolvable, f"Unresolvable value (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."
        assert count_o.value == model._count_o , f"Incorrect Result: count_o != {model._count_o}. Got: {count_o.value} at Time {get_sim_time(units='ns')}ns."

@cocotb.test
async def fuzz_test_short(dut):
    await fuzz_test(dut, 10)

@cocotb.test
async def fuzz_test_medium(dut):
    await fuzz_test(dut, 100)

@cocotb.test
async def fuzz_test_long(dut):
    await fuzz_test(dut, 1000)