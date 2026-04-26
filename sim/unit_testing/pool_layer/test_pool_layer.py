# test_pool_layer.py
import os
from   pathlib import Path
import pytest
import numpy as np

from util.utilities  import runner, lint, clock_start_sequence, reset_sequence
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from util.torch_ref  import torch_pool_ref
from functional_models.pool_layer import PoolLayerModel, RandomDataGenerator
tbpath = Path(__file__).parent

import cocotb
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

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("LineWidthPx, LineCountPx, KernelWidth, InBits, InChannels, PoolMode", 
                         [("16", "16", "2", "1", "8","0"),
                          ("16", "16", "4", "1", "9","0")])
def test_max(test_name, simulator, LineWidthPx, LineCountPx, KernelWidth, InBits, InChannels, PoolMode):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    param_str = f"InBits{InBits}_Channels{InChannels}_LineWidth{LineWidthPx}_LineCount{LineCountPx}_Kernel{KernelWidth}"
    custom_work_dir = os.path.join(tbpath, "run", "width", param_str, simulator)
    runner(simulator, timescale, tbpath, parameters, testname=test_name, work_dir=custom_work_dir)

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("LineWidthPx, LineCountPx, KernelWidth, InBits, InChannels, PoolMode", 
                         [("16", "16", "2", "1", "8", "1"),
                          ("16", "16", "4", "1", "9", "1")])
def test_avg(test_name, simulator, LineWidthPx, LineCountPx, KernelWidth, InBits, InChannels, PoolMode):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    param_str = f"InBits{InBits}_Channels{InChannels}_LineWidth{LineWidthPx}_LineCount{LineCountPx}_Kernel{KernelWidth}"
    custom_work_dir = os.path.join(tbpath, "run", "width", param_str, simulator)
    runner(simulator, timescale, tbpath, parameters, testname=test_name, work_dir=custom_work_dir)

@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("LineWidthPx, LineCountPx, KernelWidth, InBits", 
                         [("16", "16", "2", "1")])

def test_all(simulator, LineWidthPx, LineCountPx, KernelWidth, InBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    runner(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("LineWidthPx, LineCountPx, KernelWidth, InBits", [("16", "16", "2", "1")])
def test_lint(simulator, LineWidthPx, LineCountPx, KernelWidth, InBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("LineWidthPx, LineCountPx, KernelWidth, InBits", [("16", "16", "2", "1")])
def test_style(simulator, LineWidthPx, LineCountPx, KernelWidth, InBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

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

    W  = int(dut.LineWidthPx.value)
    K  = int(dut.KernelWidth.value)
    KA = K * K
    IC = int(dut.InChannels.value)
    OC = int(dut.OutChannels.value)
    S  = int(dut.Stride.value)

    # Number of accepted inputs until first valid output position (x=K-1, y=K-1)
    N_first = (K - 1) * W + (K - 1) + 1

    # We expect exactly ONE output for this test (the first valid position)
    N_out = 1
    rate = 1

    model = PoolLayerModel(dut)
    m = ModelRunner(dut, model)

    om = OutputModel(dut, RateGenerator(dut, 1), N_out)               # consume 1 output
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_first)
    dut.ready_i.value = 0
    dut.valid_i.value = 0
    dut.data_i.value = 0

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

async def rate_tests(dut, in_rate, out_rate):
    W  = int(dut.LineWidthPx.value)
    H  = int(dut.LineCountPx.value)
    K  = int(dut.KernelWidth.value)
    KA = K * K
    IC = int(dut.InChannels.value)
    OC = int(dut.OutChannels.value)
    S  = int(dut.Stride.value)
    w = int(dut.InBits.value)
    mode = int(dut.PoolMode.value)

    # Observe H rows of VALID outputs
    invalid = K - 1
    N_in = W * H
    H_out = ((H - K) // S) + 1
    W_out = ((W - K) // S) + 1
    l_out = W_out * H_out   

    input_activation  = [[[0 for _ in range(W)] for _ in range(H)] for _ in range(IC)]
    output_activation = [[[0 for _ in range(W_out)] for _ in range(H_out)] for _ in range(OC)]

    # Consumer ready probability
    slow = min(in_rate, out_rate)  # bottleneck probability
    slow = max(slow, 0.05)         # avoid insane timeouts at tiny rates in fuzz

    first_out_wait_ns = int((2 * (K - 1) * W + 2 * (K - 1) + 200) / slow)
    timeout_ns        = int((H_out * N_in + 500) / slow)
    
    model = PoolLayerModel(dut, output_activation, input_activation)
    m = ModelRunner(dut, model)

    # Consumer fuzzed; producer always drives valid
    om = OutputModel(dut, RateGenerator(dut, out_rate), l_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, in_rate), N_in)
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
        for ic in range(IC):
            print(f"\nInput Activation for IC{ic}")
            for r in range(H):
                print(" ".join(f"{input_activation[ic][r][c]:2d}" for c in range(W)))
        for oc in range(OC):
            print(f"\nOutput Activation (DUT) for OC{oc}")
            for r in range(H_out):
                print(" ".join(f"{output_activation[oc][r][c]:4d}" for c in range(W_out)))

        # Only compare against PyTorch if it is NOT 1-bit Avg Pooling
        if not (w == 1 and mode == 1):
            ref = torch_pool_ref(input_activation, kernel_size=K, stride=S, mode=mode)
            for oc in range(OC):
                print(f"\nExpected (PyTorch) for OC{oc}")
                for r in range(H_out):
                    print(" ".join(f"{int(ref[oc, r, c]):4d}" for c in range(W_out)))
            assert np.allclose(output_activation, ref.int().numpy()), "Output activation does not match PyTorch reference"

    except SimTimeoutError:
        assert 0, (
            f"Timed out. Expected {l_out} output handshakes "
            f"(W_out={W_out}, H_out={H_out}). Got {om.nproduced()} in {timeout_ns} ns. "
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
