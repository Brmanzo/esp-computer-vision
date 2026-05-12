# test_icestorm_rom.py
# Bradley Manzo 2026
import os
from   pathlib import Path
import pytest

from util.utilities  import runner, lint, clock_start_sequence, reset_sequence, \
                            auto_unpack, load_tests_from_csv

tbpath = Path(__file__).parent

import cocotb
from   cocotb.utils import get_sim_time
from   cocotb.triggers import FallingEdge, RisingEdge, Timer, Decimal
from   cocotb.types import LogicArray, Range

import random
random.seed(42)
timescale = "1ps/1ps"

tests = ['reset_test',
         'single_test',
         "write_limit_test",
         "read_all_test",
         "write_and_reset"
         ]

TEST_CASES = load_tests_from_csv(os.path.join(tbpath, "test_cases.csv"))
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)
def test_each(test_name, simulator,
              Width, Depth):
    # This line must be first
    parameters = dict(locals())
    parameters.pop('test_name', None)
    parameters.pop('simulator', None)
    runner(simulator, timescale, tbpath, parameters, testname=test_name)

# Opposite above, run all the tests in one simulation but reset
# between tests to ensure that reset is clearing all state.
@pytest.mark.parametrize("Width,Depth", [(8, 8), (11, 17)])
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
def test_all(simulator, Width, Depth):
    # This line must be first
    parameters = dict(locals())
    parameters.pop('simulator', None)
    runner(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("Width,Depth", [(11, 17)])
@pytest.mark.parametrize("simulator", ["verilator"])
def test_lint(simulator, Width, Depth):
    # This line must be first
    parameters = dict(locals())
    parameters.pop('simulator', None)
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("Width,Depth", [(11, 17)])
@pytest.mark.parametrize("simulator", ["verilator"])
def test_style(simulator, Width, Depth):
    # This line must be first
    parameters = dict(locals())
    parameters.pop('simulator', None)
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)
    
@cocotb.test
async def read_all_test(dut):
    """Test for Reading all Memory Addresses after Write"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

    wr_valid_i = dut.wr_valid_i
    wr_data_i = dut.wr_data_i
    wr_addr_i = dut.wr_addr_i
    rd_addr_i = dut.rd_addr_i
    rd_data_o = dut.rd_data_o

    depth = int(dut.Depth.value)
    width = int(dut.Width.value)
    
    wr_valid_i.value = 0
    wr_data_i.value = 0
    wr_addr_i.value = 0
    rd_addr_i.value = 0

    # Fill Memory
    mem_model = {}
    for addr in range(depth):
        await FallingEdge(clk_i)
        val = random.randint(0, (1 << width) - 1)
        wr_valid_i.value = 1
        wr_addr_i.value = addr
        wr_data_i.value = val
        mem_model[addr] = val

    await FallingEdge(clk_i)
    wr_valid_i.value = 0

    # Verify All
    for addr in range(depth):
        rd_addr_i.value = addr
        await RisingEdge(clk_i)
        await Timer(Decimal(1), units='ps') # wait for Q to settle
        assert rd_data_o.value == mem_model[addr], f"Mismatch at addr {addr}: expected {mem_model[addr]}, got {rd_data_o.value}"
        await FallingEdge(clk_i)

@cocotb.test
async def single_test(dut):
    """Write one, read one"""
    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    
    width = int(dut.Width.value)
    addr = random.randint(0, int(dut.Depth.value) - 1)
    val = random.randint(0, (1 << width) - 1)

    await FallingEdge(dut.clk_i)
    dut.wr_valid_i.value = 1
    dut.wr_addr_i.value = addr
    dut.wr_data_i.value = val
    await FallingEdge(dut.clk_i)
    dut.wr_valid_i.value = 0

    dut.rd_addr_i.value = addr
    await RisingEdge(dut.clk_i)
    await Timer(Decimal(1), units='ps')
    assert dut.rd_data_o.value == val

@cocotb.test
async def write_limit_test(dut):
    """Write at boundaries"""
    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    
    depth = int(dut.Depth.value)
    width = int(dut.Width.value)
    
    for addr in [0, depth - 1]:
        val = (1 << width) - 1
        await FallingEdge(dut.clk_i)
        dut.wr_valid_i.value = 1
        dut.wr_addr_i.value = addr
        dut.wr_data_i.value = val
        await FallingEdge(dut.clk_i)
        dut.wr_valid_i.value = 0
        dut.rd_addr_i.value = addr
        await RisingEdge(dut.clk_i)
        await Timer(Decimal(1), units='ps')
        assert dut.rd_data_o.value == val

@cocotb.test
async def write_and_reset(dut):
    """Ensure write followed by reset doesn't corrupt (logic check)"""
    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    
    width = int(dut.Width.value)
    addr = random.randint(0, int(dut.Depth.value) - 1)
    val = random.randint(0, (1 << width) - 1)

    # Write
    await FallingEdge(dut.clk_i)
    dut.wr_valid_i.value = 1
    dut.wr_addr_i.value = addr
    dut.wr_data_i.value = val
    await FallingEdge(dut.clk_i)
    dut.wr_valid_i.value = 0
    
    # Read and check
    dut.rd_addr_i.value = addr
    await RisingEdge(dut.clk_i)
    await Timer(Decimal(1), units='ps')
    assert dut.rd_data_o.value == val

    # Reset
    await reset_sequence(dut.clk_i, dut.rst_i, 5)
    
    # Read again - memory should be preserved!
    await FallingEdge(dut.clk_i)
    dut.rd_addr_i.value = addr
    await RisingEdge(dut.clk_i)
    await Timer(Decimal(1), units='ps')
    assert dut.rd_data_o.value == val