from cocotb_test import simulator
import numpy as np
import os
from   pathlib import Path
import pytest
import math

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

tests = ['reset_test'
        ,'single_test'
        ,'inout_fuzz_test'
        ,'in_fuzz_test'
        ,'out_fuzz_test'
        ,'full_bw_test']

# Format: ("Target_Var", "CSV_Column", lambda <parsed_keys_needed>: func(...))
auto_rules = [
    ("OutBits", "OutBits", lambda InBits, WeightBits, KernelWidth, InChannels, BiasBits, Unsigned: output_width(InBits, WeightBits, KernelWidth, InChannels, BiasBits, Unsigned))
]

# Format: ("Target_Var", lambda <parsed_keys_needed>: func(...))
gen_rules = [
    ("Weights", lambda WeightBits, OutChannels, InChannels, KernelWidth: gen_kernels(WeightBits, OutChannels, InChannels, KernelWidth, seed=1234)),
    ("Biases",  lambda BiasBits, OutChannels: gen_biases(BiasBits, OutChannels, seed=1234))
]

def run_conv_test(test_name, simulator, parameters, Weights, Biases, test_class):
    InBits      = int(parameters["InBits"])
    WeightBits  = int(parameters["WeightBits"])
    OutBits     = int(parameters["OutBits"])
    KernelWidth = int(parameters["KernelWidth"])
    InChannels  = int(parameters["InChannels"])
    OutChannels = int(parameters["OutChannels"])
    BiasBits    = int(parameters["BiasBits"])
    DSPCount    = int(parameters.get("DSPCount", 0))

    if simulator == "icarus" and DSPCount > 0:
        pytest.skip("Icarus Verilog has issues with ROM initialization in sequential configurations")

    if DSPCount > 0:
        req_bits = max(OutBits, WeightBits + InBits + math.ceil(math.log2(InChannels * (KernelWidth**2))))
        if req_bits > 32:
            pytest.skip(f"conv_layer DSP accumulator (32 bits) too small for required {req_bits} bits")

    param_str = f"IC_{InChannels}_OC_{OutChannels}_test_{test_name}"
    if test_class == "width":
        param_str = f"InBits_{InBits}_WeightBits_{WeightBits}_OutBits_{OutBits}_test_{test_name}"
    elif test_class == "stride":
        param_str = f"KW_{KernelWidth}_S_{parameters['Stride']}_test_{test_name}"
    elif test_class == "padding":
        param_str = f"KW_{KernelWidth}_P_{parameters['Padding']}_test_{test_name}"

    weight_bits  = WeightBits
    bias_bits    = OutChannels * BiasBits
    weight_count = OutChannels * InChannels * (KernelWidth**2)

    # Remove injected params so cocotb-runner doesn't pass them on CLI
    clean_params = parameters.copy()
    clean_params.pop("Weights", None)
    clean_params.pop("Biases", None)

    custom_work_dir = inject_weights_and_biases(
        simulator=simulator, parameters=clean_params, param_str=param_str, 
        tbpath=tbpath, test_class=test_class, Weights=Weights, Biases=Biases, 
        weight_bits=weight_bits, bias_bits=bias_bits, weight_count=weight_count,
        layer=0, dsp_count=DSPCount)

    filelist = "filelists/conv_layer.json"
    runner(
        simulator=simulator, timescale=timescale, tbpath=tbpath, params=clean_params, 
        testname=test_name, work_dir=custom_work_dir, includes=[custom_work_dir],
        toplevel_override="tb_conv_layer", extra_sources=[os.path.join(tbpath, "tb_conv_layer.sv")],
        filelist=filelist
    )

TEST_CASES_WIDTH = load_tests_from_csv(os.path.join(tbpath, "test_cases_width.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES_WIDTH)
def test_width(test_name, simulator,
               InBits, WeightBits, OutBits, KernelWidth, LineWidthPx, Weights, Biases,
               LineCountPx, InChannels, OutChannels, BiasBits, Stride, Padding, DSPCount, Unsigned):
    run_conv_test(test_name, simulator, locals(), Weights, Biases, "width")

TEST_CASES_STRIDE = load_tests_from_csv(os.path.join(tbpath, "test_cases_stride.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES_STRIDE)
def test_stride(test_name, simulator,
                InBits, WeightBits, OutBits, KernelWidth, LineWidthPx, Weights, Biases,
                LineCountPx, InChannels, OutChannels, BiasBits, Stride, Padding, DSPCount, Unsigned):
    run_conv_test(test_name, simulator, locals(), Weights, Biases, "stride")

TEST_CASES_PADDING = load_tests_from_csv(os.path.join(tbpath, "test_cases_padding.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES_PADDING)
def test_padding(test_name, simulator,
                 InBits, WeightBits, OutBits, KernelWidth, LineWidthPx, Weights, Biases,
                 LineCountPx, InChannels, OutChannels, BiasBits, Stride, Padding, DSPCount, Unsigned):
    run_conv_test(test_name, simulator, locals(), Weights, Biases, "padding")

TEST_CASES_CHANNELS = load_tests_from_csv(os.path.join(tbpath, "test_cases_channels.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES_CHANNELS)
def test_channels(test_name, simulator,
                  InBits, WeightBits, OutBits, KernelWidth, LineWidthPx, Weights, Biases,
                  LineCountPx, InChannels, OutChannels, BiasBits, Stride, Padding, DSPCount, Unsigned):
    run_conv_test(test_name, simulator, locals(), Weights, Biases, "channels")

TEST_CASES_DSPS = load_tests_from_csv(os.path.join(tbpath, "test_cases_dsps.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES_DSPS)
def test_dsps(test_name, simulator,
              InBits, WeightBits, OutBits, KernelWidth, LineWidthPx, Weights, Biases,
              LineCountPx, InChannels, OutChannels, BiasBits, Stride, Padding, DSPCount, Unsigned):
    run_conv_test(test_name, simulator, locals(), Weights, Biases, "dsps")

TEST_CASES_OUT_ACT = load_tests_from_csv(os.path.join(tbpath, "test_cases_out_act.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES_OUT_ACT)
def test_out_act(test_name, simulator,
                 InBits, WeightBits, OutBits, KernelWidth, LineWidthPx, Weights, Biases,
                 LineCountPx, InChannels, OutChannels, BiasBits, Stride, Padding, DSPCount, ShiftBits, Unsigned):
    run_conv_test(test_name, simulator, locals(), Weights, Biases, "out_act")

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("DSPCount", [0, 1, 2, 4])
@pytest.mark.parametrize("LineWidthPx, InBits, OutBits", [("16", "1", output_width(1, 2, 3, 1))])
def test_lint(simulator, LineWidthPx, InBits, OutBits, DSPCount):
    parameters = dict(locals())
    del parameters['simulator']
    filelist = "filelists/conv_layer.json"
    lint(simulator, timescale, tbpath, parameters, filelist=filelist)

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
    unsigned_obj = getattr(dut, "Unsigned", None)
    Unsigned = int(unsigned_obj.value) if unsigned_obj is not None else 0

    real_rows_before_out = max(0, (K - 1) - P)
    real_pixels_in_out_row = max(0, (K - 1) - P + 1)
    N_first = (real_rows_before_out * W) + real_pixels_in_out_row
    N_out = 1
    rate = 1

    packed_weights = int(os.environ["INJECTED_WEIGHTS_0_INT"], 0)
    kernels_4d = unpack_kernel_weights(packed_weights, WW, OC, IC, K)
    packed_biases = int(os.environ.get("INJECTED_BIASES_0_INT", "0"), 0)
    biases_2d = unpack_biases(packed_biases, BW, OC)
    
    model = ConvLayerModel(dut, weights=kernels_4d, biases=biases_2d)
    m = ModelRunner(dut, model)

    om = OutputModel(dut, RateGenerator(dut, 1), N_out)
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

    # Calculate sequential latency
    cycles_per_vec = 4
    if hasattr(dut, "DSPCount") and int(dut.DSPCount.value) > 0:
        eff_dsps = min(int(dut.DSPCount.value), OC)
        neurons_per_dsp = (OC + eff_dsps - 1) // eff_dsps
        cycles_per_vec = (IC * K * K + 4) * neurons_per_dsp + 10
    
    tmo_ns = cycles_per_vec * N_first + 5000

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
    unsigned_obj = getattr(dut, "Unsigned", None)
    Unsigned = int(unsigned_obj.value) if unsigned_obj is not None else 0

    invalid = K - 1
    N_in = W * H
    P_W = W + 2 * P
    P_H = H + 2 * P

    H_out = ((P_H - K) // S) + 1
    W_out = ((P_W - K) // S) + 1
    l_out = W_out * H_out   

    input_activation  = [[[0 for _ in range(W)] for _ in range(H)] for _ in range(IC)]
    output_activation = [[[0 for _ in range(W_out)] for _ in range(H_out)] for _ in range(OC)]

    slow = min(in_rate, out_rate)
    slow = max(slow, 0.05)

    # Scale timeout for sequential DSP implementation
    scale = 1
    if hasattr(dut, "DSPCount") and int(dut.DSPCount.value) > 0:
        eff_dsps = min(int(dut.DSPCount.value), OC)
        neurons_per_dsp = (OC + eff_dsps - 1) // eff_dsps
        scale = (IC * K * K + 4) * neurons_per_dsp + 10

    first_out_wait_ns = int((2 * (K - 1) * W + 2 * (K - 1) + 200) * scale / slow)
    timeout_ns        = int((H_out * N_in + 5000) * scale / slow)

    packed_weights = int(os.environ["INJECTED_WEIGHTS_0_INT"], 0)
    packed_biases  = int(os.environ.get("INJECTED_BIASES_0_INT", "0"), 0)

    kernels_4d = unpack_kernel_weights(packed_weights, WW, OC, IC, K)
    biases_2d  = unpack_biases(packed_biases, BW, OC)
    model = ConvLayerModel(dut, weights=kernels_4d, output_activation=output_activation, input_activation=input_activation, biases=biases_2d)
    m = ModelRunner(dut, model)

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

    try:
        await with_timeout(RisingEdge(dut.valid_o), first_out_wait_ns, 'ns')
    except SimTimeoutError:
        assert False, (
            f"Timed out waiting for valid_o high. "
            f"W={W}, K={K}, S={S}, H_out={H_out}, W_out={W_out}, N_in={N_in}, waited={first_out_wait_ns} ns."
        )

    try:
        await om.wait(timeout_ns)
        if sim_verbose():
            for ic in range(IC):
                print(f"\nInput Activation for IC{ic}")
                for r in range(H):
                    print(" ".join(f"{input_activation[ic][r][c]:2d}" for c in range(W)))

            for oc in range(OC):
                print(f"\nKernel for OC{oc}")
                for ic in range(IC):
                    print(f"  IC{ic}")
                    for r in range(K):
                        print(" ".join(f"{kernels_4d[oc][ic][r][c]:4d}" for c in range(K)))

            for oc in range(OC):
                print(f"\nOutput Activation (DUT) for OC{oc}")
                for r in range(H_out):
                    print(" ".join(f"{output_activation[oc][r][c]:4d}" for c in range(W_out)))

        shift_obj = getattr(dut, "ShiftBits", None)
        shift_bits = int(shift_obj.value) if shift_obj is not None else 0
        from functional_models.conv_layer import calc_acc_bits
        acc_bits = calc_acc_bits(K, int(dut.InBits.value), WW, IC, BW, Unsigned)
        ref = torch_conv_ref(
            input_activation, kernels_4d, S,
            in_bits=int(dut.InBits.value), out_bits=int(dut.OutBits.value),
            padding=int(dut.Padding.value), biases=biases_2d,
            shift_bits=shift_bits, acc_bits=acc_bits, unsigned=Unsigned
        )
        
        if sim_verbose():
            for oc in range(OC):
                print(f"\nExpected (PyTorch) for OC{oc}")
                for r in range(H_out):
                    print(" ".join(f"{int(ref[oc, r, c]):4d}" for c in range(W_out)))

        assert np.allclose(output_activation, ref.int().numpy()), "Output activation does not match PyTorch reference"
    except SimTimeoutError:
        assert False, (
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
