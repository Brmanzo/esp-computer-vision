
from   pathlib import Path
import pytest

from util.utilities  import runner, lint, clock_start_sequence, reset_sequence

tbpath = Path(__file__).parent

import cocotb
from   cocotb.utils import get_sim_time
from   cocotb.triggers import FallingEdge
from   cocotb.types import LogicArray, Range, LogicArray

import random
random.seed(42)
timescale = "1ps/1ps"

tests = ['reset_test',
         'single_test',
         "write_limit_test",
         "read_all_test",
         "write_and_reset",
         "conflict_test"
         ]

@pytest.mark.parametrize("Width,Depth", [(8, 8), (11, 17)])
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
def test_each(test_name, simulator, Width, Depth):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters, testname=test_name)

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("Width,Depth", [(8, 8), (11, 17)])
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
def test_all(simulator, Width, Depth):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("Width,Depth", [(11, 17)])
@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator, Width, Depth):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("Width,Depth", [(11, 17)])
@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator, Width, Depth):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)
    
@cocotb.test
async def read_all_test(dut):
    """Test for Reading all Memory Addresses"""

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    wr_valid_i = dut.wr_valid_i
    wr_data_i = dut.wr_data_i
    wr_addr_i = dut.wr_addr_i

    rd_data_o = dut.rd_data_o
    rd_addr_i = dut.rd_addr_i
    rd_valid_i = dut.rd_valid_i

    inputs = [wr_valid_i, wr_data_i, wr_addr_i, rd_addr_i, rd_valid_i]

    depth = dut.Depth.value
    width = dut.Width.value
    
    for p in inputs:
        p.value = 0

    await FallingEdge(clk_i)

    for i in range(0, dut.Depth.value):
        wr_addr_i.value = i
        wr_valid_i.value = 1
        s = bin(i)[2:]
        s = '0'* (width-len(s)) + s
        wr_data_i.value = LogicArray(s[0:width], Range(width-1,'downto', 0))

        await FallingEdge(clk_i)

    wr_addr_i.value = 0
    wr_valid_i.value = 0

    await FallingEdge(clk_i)

    for i in range(0, dut.Depth.value):
        rd_addr_i.value = i
        rd_valid_i.value = 1

        await FallingEdge(clk_i)
        s = bin(i)[2:]
        s = '0'* (width-len(s)) + s
        expected = LogicArray(s[0:width], Range(width-1,'downto', 0)).integer

        assert rd_data_o.value.is_resolvable, f"Unresolvable value (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."
        assert rd_data_o.value == expected, f"Incorrect result read from memory at index {i}: Expected {expected}. Got: {rd_data_o.value} at Time {get_sim_time(units='ns')}ns."


@cocotb.test
async def single_test(dut):
    """Test for writing a single address"""

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    wr_valid_i = dut.wr_valid_i
    wr_data_i = dut.wr_data_i
    wr_addr_i = dut.wr_addr_i

    rd_data_o = dut.rd_data_o
    rd_addr_i = dut.rd_addr_i
    rd_valid_i = dut.rd_valid_i

    inputs = [wr_valid_i, wr_data_i, wr_addr_i, rd_addr_i, rd_valid_i]

    depth = dut.Depth.value
    width = dut.Width.value

    for p in inputs:
        p.value = 0

    await FallingEdge(clk_i)

    s = '0110' * width
    wr_addr_i.value = 0
    wr_data_i.value = LogicArray(s[0:width], Range(width-1,'downto', 0))
    wr_valid_i.value = 1

    await FallingEdge(clk_i)

    wr_addr_i.value = 0
    wr_data_i.value = 0
    wr_valid_i.value = 0

    rd_addr_i.value = 0
    rd_valid_i.value = 1
    expected = LogicArray(s[0:width], Range(width-1, 'downto', 0)).integer

    await FallingEdge(clk_i)
    
    assert rd_data_o.value.is_resolvable, f"Unresolvable value (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."
    assert rd_data_o.value == expected, f"Incorrect result read back from address 0: Expected {expected}. Got: {rd_data_o.value} at Time {get_sim_time(units='ns')}ns."

    rd_addr_i.value = 0
    rd_valid_i.value = 0

@cocotb.test
async def write_limit_test(dut):
    """Test for the first and last addresses"""

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    wr_valid_i = dut.wr_valid_i
    wr_data_i = dut.wr_data_i
    wr_addr_i = dut.wr_addr_i

    rd_data_o = dut.rd_data_o
    rd_addr_i = dut.rd_addr_i
    rd_valid_i = dut.rd_valid_i

    inputs = [wr_valid_i, wr_data_i, wr_addr_i, rd_addr_i, rd_valid_i]

    depth = dut.Depth.value
    width = dut.Width.value

    for p in inputs:
        p.value = 0

    await FallingEdge(clk_i)

    s = '01' * width
    wr_addr_i.value = 0
    wr_data_i.value = LogicArray(s[0:width], Range(width-1,'downto', 0))
    wr_valid_i.value = 1

    await FallingEdge(clk_i)

    wr_addr_i.value = depth - 1
    wr_data_i.value = LogicArray(s[1:width + 1], Range(width-1,'downto', 0))
    wr_valid_i.value = 1

    await FallingEdge(clk_i)

    wr_addr_i.value = 0
    wr_data_i.value = 0
    wr_valid_i.value = 0

    # Wait a bit, to see what happens.
    await FallingEdge(clk_i)
    await FallingEdge(clk_i)
    await FallingEdge(clk_i)

    rd_addr_i.value = 0
    rd_valid_i.value = 1
    expected = LogicArray(s[0:width], Range(width-1, 'downto', 0)).integer

    await FallingEdge(clk_i)
    
    assert rd_data_o.value.is_resolvable, f"Unresolvable value (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."
    assert rd_data_o.value == expected, f"Incorrect result read back from address 0: Expected {expected}. Got: {rd_data_o.value} at Time {get_sim_time(units='ns')}ns."

    rd_addr_i.value = depth-1
    rd_valid_i.value = 1
    expected = LogicArray(s[1:width + 1], Range(width-1,'downto', 0)).integer

    await FallingEdge(clk_i)
    
    assert rd_data_o.value.is_resolvable, f"Unresolvable value (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."
    assert rd_data_o.value == expected, f"Incorrect result read from address {depth-1}: Expected {expected}. Got: {rd_data_o.value} at Time {get_sim_time(units='ns')}ns."

@cocotb.test
async def write_and_reset(dut):
    """Test for writing while under reset"""

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    wr_valid_i = dut.wr_valid_i
    wr_data_i = dut.wr_data_i
    wr_addr_i = dut.wr_addr_i

    rd_data_o = dut.rd_data_o
    rd_addr_i = dut.rd_addr_i
    rd_valid_i = dut.rd_valid_i

    inputs = [wr_valid_i, wr_data_i, wr_addr_i, rd_addr_i, rd_valid_i]

    depth = dut.Depth.value
    width = dut.Width.value

    for p in inputs:
        p.value = 0

    await FallingEdge(clk_i)

    s = '11' * width
    wr_addr_i.value = 1
    wr_data_i.value = LogicArray(s[0:width], Range(width - 1,'downto', 0))
    wr_valid_i.value = 1

    await FallingEdge(clk_i)

    wr_addr_i.value = 0
    wr_data_i.value = 0
    wr_valid_i.value = 0

    # Wait a bit, to see what happens.
    await FallingEdge(clk_i)
    await FallingEdge(clk_i)
    await FallingEdge(clk_i)

    s = '0' * width
    wr_addr_i.value = 1
    rst_i.value = 1
    wr_data_i.value = LogicArray(s[0:width], Range(width - 1,'downto', 0))
    wr_valid_i.value = 1

    await FallingEdge(clk_i)

    wr_addr_i.value = 1
    rst_i.value = 0
    wr_data_i.value = LogicArray(s[0:width], Range(width - 1,'downto', 0))
    wr_valid_i.value = 0

    await FallingEdge(clk_i)
    await FallingEdge(clk_i)
    await FallingEdge(clk_i)

    rd_addr_i.value = 1
    rd_valid_i.value = 1
    s = '1' * width
    expected = LogicArray(s[0:width], Range(width-1, 'downto', 0)).integer

    await FallingEdge(clk_i)
    
    assert rd_data_o.value.is_resolvable, f"Unresolvable value (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."
    assert rd_data_o.value == expected, f"Incorrect result. Writing during reset should not affect memory contents. Expected {expected}. Got: {rd_data_o.value} at Time {get_sim_time(units='ns')}ns."


@cocotb.test
async def conflict_test(dut):
    """Test for writing and reading the same address"""

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    wr_valid_i = dut.wr_valid_i
    wr_data_i = dut.wr_data_i
    wr_addr_i = dut.wr_addr_i

    rd_data_o = dut.rd_data_o
    rd_addr_i = dut.rd_addr_i
    rd_valid_i = dut.rd_valid_i

    inputs = [wr_valid_i, wr_data_i, wr_addr_i, rd_addr_i, rd_valid_i]

    depth = dut.Depth.value
    width = dut.Width.value

    for p in inputs:
        p.value = 0

    await FallingEdge(clk_i)

    # Write address 1
    s = '11' * width
    wr_addr_i.value = 1
    wr_data_i.value = LogicArray(s[0:width], Range(width - 1,'downto', 0))
    wr_valid_i.value = 1

    await FallingEdge(clk_i)

    wr_addr_i.value = 0
    wr_data_i.value = 0
    wr_valid_i.value = 0

    # Wait a bit, to see what happens.
    await FallingEdge(clk_i)
    await FallingEdge(clk_i)
    await FallingEdge(clk_i)

    # Write address 1 again, with a different value
    # also read.
    s = '0' * width
    wr_addr_i.value = 1
    rd_addr_i.value = 1
    rd_valid_i.value = 1
    wr_data_i.value = LogicArray(s[0:width], Range(width - 1,'downto', 0))
    wr_valid_i.value = 1

    await FallingEdge(clk_i)

    wr_valid_i.value = 0
    # Old value
    s = '1' * width
    expected = LogicArray(s[0:width], Range(width-1, 'downto', 0)).integer
    
    assert rd_data_o.value.is_resolvable, f"Unresolvable value (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."
    assert rd_data_o.value == expected, f"Incorrect result. Reading while writing the same address should return old data after one cycle. Expected {expected}. Got: {rd_data_o.value} at Time {get_sim_time(units='ns')}ns."

    await FallingEdge(clk_i)

    # New value
    s = '0' * width
    expected = LogicArray(s[0:width], Range(width-1, 'downto', 0)).integer

    assert rd_data_o.value.is_resolvable, f"Unresolvable value (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."
    assert rd_data_o.value == expected, f"Incorrect result. Reading while writing the same address should return new data after two cycles. Expected {expected}. Got: {rd_data_o.value} at Time {get_sim_time(units='ns')}ns."