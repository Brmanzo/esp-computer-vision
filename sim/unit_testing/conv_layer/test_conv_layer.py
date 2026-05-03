from cocotb_test import simulator
import numpy as np
import os
from   pathlib import Path
import pytest

from util.utilities  import inject_weights_and_biases, runner, lint, \
                            sim_verbose, clock_start_sequence, reset_sequence, \
                            load_tests_from_csv, auto_unpack
from util.bitwise    import unpack_kernel_weights, unpack_biases
from util.gen_inputs import gen_kernels, gen_biases
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from functional_models.conv_layer import ConvLayerModel, RandomDataGenerator, output_width
from util.torch_ref import torch_conv_ref
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

# Format: ("Target_Var", "CSV_Column", lambda <parsed_keys_needed>: func(...))
auto_rules = [
    ("OutBits", "OutBits", lambda InBits, WeightBits, KernelWidth, InChannels, BiasBits: output_width(InBits, WeightBits, KernelWidth, InChannels, BiasBits))
]

# Format: ("Target_Var", lambda <parsed_keys_needed>: func(...))
gen_rules = [
    ("Weights", lambda WeightBits, OutChannels, InChannels, KernelWidth: gen_kernels(WeightBits, OutChannels, InChannels, KernelWidth, seed=1234)),
    ("Biases",  lambda BiasBits, OutChannels: gen_biases(BiasBits, OutChannels, seed=1234))
]

TEST_CASES_WIDTH = load_tests_from_csv(os.path.join(tbpath, "test_cases_width.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES_WIDTH)
def test_width(test_name, simulator,
               InBits, WeightBits, OutBits, KernelWidth, LineWidthPx, Weights, Biases,
               LineCountPx, InChannels, OutChannels, BiasBits, Stride, Padding):
    parameters = dict(locals())
    parameters.pop("Weights", None)
    parameters.pop("Biases", None)
    param_str = f"InBits_{InBits}_WeightBits_{WeightBits}_OutBits_{OutBits}_test_{test_name}"
    
    weight_bits = OutChannels * InChannels * (KernelWidth**2) * WeightBits
    bias_bits   = OutChannels * BiasBits

    custom_work_dir = inject_weights_and_biases(
        simulator=simulator, parameters=parameters, param_str=param_str, 
        tbpath=tbpath, test_class="width", Weights=Weights, Biases=Biases, 
        weight_bits=weight_bits, bias_bits=bias_bits)

    runner(
        simulator=simulator, timescale=timescale, tbpath=tbpath, params=parameters, 
        testname=test_name, work_dir=custom_work_dir, includes=[custom_work_dir],
        toplevel_override="tb_conv_layer", extra_sources=[os.path.join(tbpath, "tb_conv_layer.sv")]
    )
TEST_CASES_STRIDE = load_tests_from_csv(os.path.join(tbpath, "test_cases_stride.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES_STRIDE)
def test_stride(test_name, simulator,
                InBits, WeightBits, OutBits, KernelWidth, LineWidthPx, Weights, Biases,
                LineCountPx, InChannels, OutChannels, BiasBits, Stride, Padding):
    parameters = dict(locals())
    parameters.pop("Weights", None)
    parameters.pop("Biases", None)
    param_str = f"KW_{KernelWidth}_S_{Stride}_test_{test_name}"

    weight_bits = OutChannels * InChannels * (KernelWidth**2) * WeightBits
    bias_bits   = OutChannels * BiasBits

    custom_work_dir = inject_weights_and_biases(
        simulator=simulator, parameters=parameters, param_str=param_str, 
        tbpath=tbpath, test_class="stride", Weights=Weights, Biases=Biases, 
        weight_bits=weight_bits, bias_bits=bias_bits)
    
    runner(
        simulator=simulator, timescale=timescale, tbpath=tbpath, params=parameters, 
        testname=test_name, work_dir=custom_work_dir, includes=[custom_work_dir],
        toplevel_override="tb_conv_layer", extra_sources=[os.path.join(tbpath, "tb_conv_layer.sv")]
    )

TEST_CASES_PADDING = load_tests_from_csv(os.path.join(tbpath, "test_cases_padding.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES_PADDING)
def test_padding(test_name, simulator,
                InBits, WeightBits, OutBits, KernelWidth, LineWidthPx, Weights, Biases,
                LineCountPx, InChannels, OutChannels, BiasBits, Stride, Padding):
    parameters = dict(locals())
    parameters.pop("Weights", None)
    parameters.pop("Biases", None)
    param_str = f"KW_{KernelWidth}_P_{Padding}_test_{test_name}"

    weight_bits = OutChannels * InChannels * (KernelWidth**2) * WeightBits
    bias_bits   = OutChannels * BiasBits

    custom_work_dir = inject_weights_and_biases(
        simulator=simulator, parameters=parameters, param_str=param_str, 
        tbpath=tbpath, test_class="padding", Weights=Weights, Biases=Biases, 
        weight_bits=weight_bits, bias_bits=bias_bits)
    
    runner(
        simulator=simulator, timescale=timescale, tbpath=tbpath, params=parameters, 
        testname=test_name, work_dir=custom_work_dir, includes=[custom_work_dir],
        toplevel_override="tb_conv_layer", extra_sources=[os.path.join(tbpath, "tb_conv_layer.sv")]
    )
TEST_CASES_CHANNELS = load_tests_from_csv(os.path.join(tbpath, "test_cases_channels.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES_CHANNELS)
def test_channels(test_name, simulator,
                InBits, WeightBits, OutBits, KernelWidth, LineWidthPx, Weights, Biases,
                LineCountPx, InChannels, OutChannels, BiasBits, Stride, Padding):
    parameters = dict(locals())
    parameters.pop("Weights", None)
    parameters.pop("Biases", None)
    param_str = f"IC_{InChannels}_OC_{OutChannels}_test_{test_name}"

    weight_bits = OutChannels * InChannels * (KernelWidth**2) * WeightBits
    bias_bits   = OutChannels * BiasBits
   
    custom_work_dir = inject_weights_and_biases(
        simulator=simulator, parameters=parameters, param_str=param_str, 
        tbpath=tbpath, test_class="channels", Weights=Weights, Biases=Biases, 
        weight_bits=weight_bits, bias_bits=bias_bits)

    runner(
        simulator=simulator, timescale=timescale, tbpath=tbpath, params=parameters, 
        testname=test_name, work_dir=custom_work_dir, includes=[custom_work_dir],
        toplevel_override="tb_conv_layer", extra_sources=[os.path.join(tbpath, "tb_conv_layer.sv")]
    )

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("LineWidthPx, InBits, OutBits", [("16", "1", output_width(1, 2, 3, 1))])
def test_lint(simulator, LineWidthPx, InBits, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("LineWidthPx, InBits, OutBits", [("16", "1", output_width(1, 2, 3, 1))])
def test_style(simulator, LineWidthPx, InBits, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

@cocotb.test
async def single_test(dut):
    """Drive pixels until the first VALID kernel position, then expect 1 output."""

    W  = int(dut.LineWidthPx.value)
    K  = int(dut.KernelWidth.value)
    IC = int(dut.InChannels.value)
    OC = int(dut.OutChannels.value)
    BW = int(dut.BiasBits.value)
    WW = int(dut.WeightBits.value)
    P  = int(dut.Padding.value)
    S  = int(dut.Stride.value)

    # Calculate how many full rows of REAL data are consumed before the output row
    real_rows_before_out = max(0, (K - 1) - P)
    
    # Calculate how many REAL pixels are consumed in the final row to reach the window edge
    real_pixels_in_out_row = max(0, (K - 1) - P + 1)

    # Number of accepted inputs until first valid output position
    N_first = (real_rows_before_out * W) + real_pixels_in_out_row

    # We expect exactly ONE output for this test (the first valid position)
    N_out = 1

    rate = 1

    packed_weights = int(os.environ["INJECTED_WEIGHTS_0_INT"])
    kernels_4d = unpack_kernel_weights(packed_weights, WW, OC, IC, K)

    packed_biases = int(os.environ.get("INJECTED_BIASES_0_INT", "0"))
    biases_2d = unpack_biases(packed_biases, BW, OC)
    
    model = ConvLayerModel(dut, weights=kernels_4d, biases=biases_2d)
    m = ModelRunner(dut, model)

    om = OutputModel(dut, RateGenerator(dut, 1), N_out)               # consume 1 output
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_first)  # produce N_first inputs

    dut.ready_i.value = 0
    dut.valid_i.value = 0
    dut.data_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait until that single output is observed
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
    IC = int(dut.InChannels.value)
    OC = int(dut.OutChannels.value)
    WW = int(dut.WeightBits.value)
    BW = int(dut.BiasBits.value)
    S  = int(dut.Stride.value)
    P  = int(dut.Padding.value)

    # Observe H rows of VALID outputs
    invalid = K - 1
    N_in = W * H

    P_W = W + 2 * P
    P_H = H + 2 * P

    H_out = ((P_H - K) // S) + 1
    W_out = ((P_W - K) // S) + 1
    l_out = W_out * H_out   

    input_activation  = [[[0 for _ in range(W)] for _ in range(H)] for _ in range(IC)]
    output_activation = [[[0 for _ in range(W_out)] for _ in range(H_out)] for _ in range(OC)]

    # Consumer ready probability
    slow = min(in_rate, out_rate)  # bottleneck probability
    slow = max(slow, 0.05)         # avoid insane timeouts at tiny rates in fuzz

    first_out_wait_ns = int((2 * (K - 1) * W + 2 * (K - 1) + 200) / slow)
    timeout_ns        = int((H_out * N_in + 500) / slow)

    packed_weights = int(os.environ["INJECTED_WEIGHTS_0_INT"])
    packed_biases  = int(os.environ.get("INJECTED_BIASES_0_INT", "0"))

    kernels_4d = unpack_kernel_weights(packed_weights, WW, OC, IC, K)
    biases_2d  = unpack_biases(packed_biases, BW, OC)
    model = ConvLayerModel(dut, weights=kernels_4d, output_activation=output_activation, input_activation=input_activation, biases=biases_2d)
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
        # --- Print input ---
        if sim_verbose():
            for ic in range(IC):
                print(f"\nInput Activation for IC{ic}")
                for r in range(H):
                    print(" ".join(f"{input_activation[ic][r][c]:2d}" for c in range(W)))

            # --- Print kernels ---
            for oc in range(OC):
                print(f"\nKernel for OC{oc}")
                for ic in range(IC):
                    print(f"  IC{ic}")
                    for r in range(K):
                        print(" ".join(f"{kernels_4d[oc][ic][r][c]:4d}" for c in range(K)))

            # --- Print DUT-captured output (make sure output_activation is H_out x W_out) ---
            for oc in range(OC):
                print(f"\nOutput Activation (DUT) for OC{oc}")
                for r in range(H_out):
                    print(" ".join(f"{output_activation[oc][r][c]:4d}" for c in range(W_out)))

        # Verify against PyTorch reference convolution
        ref = torch_conv_ref(
            input_activation, 
            kernels_4d, 
            S, 
            in_bits=int(dut.InBits.value), 
            out_bits=int(dut.OutBits.value), 
            padding=int(dut.Padding.value),
            biases=biases_2d
        )
        
        if sim_verbose():
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
