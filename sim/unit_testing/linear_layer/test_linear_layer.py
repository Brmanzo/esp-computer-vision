# test_linear_layer.py
import os
from   pathlib import Path
import pytest
import shutil

from util.utilities  import runner, lint, clock_start_sequence, reset_sequence
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from util.bitwise    import unpack_weights, unpack_biases
from util.gen_inputs import gen_weights, gen_biases
from functional_models.linear_layer import LinearLayerModel, RandomDataGenerator, output_width
from util.torch_ref import torch_linear_ref
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import FallingEdge
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
@pytest.mark.parametrize(
    "InBits, WeightBits, InChannels, OutBits, OutChannels, Weights, BiasBits, Biases",
    [
        (1, 2, 1, output_width(1, 2, 1), 1, gen_weights(2, 1, 1, seed=1234), 2, gen_biases(2, 1)),
        (2, 3, 1, output_width(2, 3, 1), 1, gen_weights(3, 1, 1, seed=1234), 3, gen_biases(3, 1)),
        (4, 5, 1, output_width(4, 5, 1), 1, gen_weights(5, 1, 1, seed=1234), 5 ,gen_biases(5 ,1)),
        (8, 8 ,1 ,output_width(8 ,8 ,1) ,1 ,gen_weights(8 ,1 ,1 ,seed=1234) ,8 ,gen_biases(8 ,1)),
    ],
)
def test_width(test_name, simulator, InBits, WeightBits, InChannels, OutBits, OutChannels, Weights, BiasBits, Biases):
    parameters = dict(locals())
    del parameters["test_name"]
    del parameters["simulator"]

    # Remove injected params so cocotb-runner doesn't pass them on CLI
    del parameters["Weights"]
    del parameters["Biases"]

    param_str = f"InBits_{InBits}_WeightBits_{WeightBits}_OutBits_{OutBits}_BiasBits_{BiasBits}_test_{test_name}"
    custom_work_dir = os.path.join(tbpath, "run", "width", param_str, simulator)
    if simulator.startswith("icarus") and os.path.exists(custom_work_dir):
        shutil.rmtree(custom_work_dir)
    os.makedirs(custom_work_dir, exist_ok=True)

    # ---- Emit injected_weights.vh ----
    total_bits_w = int(OutChannels) * int(InChannels) * int(WeightBits)
    vh_path = os.path.join(custom_work_dir, "injected_weights.vh")
    with open(vh_path, "w") as f:
        hex_width = (total_bits_w + 3) // 4
        f.write(
            f"localparam logic signed [{total_bits_w-1}:0] INJECTED_WEIGHTS = "
            f"{total_bits_w}'h{Weights:0{hex_width}x};\n"
        )

    total_bits_b = int(OutChannels) * int(BiasBits)
    vhb_path = os.path.join(custom_work_dir, "injected_biases.vh")
    with open(vhb_path, "w") as f:
        hex_width = (total_bits_b + 3) // 4
        f.write(
            f"localparam logic signed [{total_bits_b-1}:0] INJECTED_BIASES = "
            f"{total_bits_b}'h{Biases:0{hex_width}x};\n"
        )

    # ---- Pass big ints via env vars for cocotb ----
    os.environ["INJECTED_WEIGHTS_INT"] = str(Weights)
    os.environ["INJECTED_BIASES_INT"]  = str(Biases)

    wrapper_path = os.path.join(tbpath, "tb_linear_layer.sv")

    runner(
        simulator=simulator,
        timescale=timescale,
        tbpath=tbpath,
        params=parameters,
        testname=test_name,
        work_dir=custom_work_dir,
        includes=[custom_work_dir],        # so injected_*.vh can be `included
        toplevel_override="tb_linear_layer",
        extra_sources=[wrapper_path],
    )

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize(
    "InBits, WeightBits, InChannels, OutBits, OutChannels, Weights, BiasBits, Biases",
    [
        (1, 2,  2, output_width(1, 2, 1),    1, gen_weights(2,  1,  1, seed=1234), 2, gen_biases(2, 1)),
        (2, 3,  2, output_width(2, 3, 2),    2, gen_weights(3,  2,  2, seed=1234), 3, gen_biases(3, 2)),
        (4, 5,  4, output_width(4, 5, 4),    4, gen_weights(5,  4,  4, seed=1234), 5, gen_biases(5, 4)),
        (8, 8, 32, output_width(8, 8, 32),  32, gen_weights(8, 32, 32, seed=1234), 8, gen_biases(8, 32)),
    ],
)
def test_channels(test_name, simulator, InBits, WeightBits, InChannels, OutBits, OutChannels, Weights, BiasBits, Biases):
    parameters = dict(locals())
    del parameters["test_name"]
    del parameters["simulator"]

    # Remove injected params so cocotb-runner doesn't pass them on CLI
    del parameters["Weights"]
    del parameters["Biases"]

    param_str = f"InChannels_{InChannels}_OutChannels_{OutChannels}_test_{test_name}"
    custom_work_dir = os.path.join(tbpath, "run", "channels", param_str, simulator)
    if simulator.startswith("icarus") and os.path.exists(custom_work_dir):
        shutil.rmtree(custom_work_dir)
    os.makedirs(custom_work_dir, exist_ok=True)

    # ---- Emit injected_weights.vh ----
    total_bits_w = int(OutChannels) * int(InChannels) * int(WeightBits)
    vh_path = os.path.join(custom_work_dir, "injected_weights.vh")
    with open(vh_path, "w") as f:
        hex_width = (total_bits_w + 3) // 4
        f.write(
            f"localparam logic signed [{total_bits_w-1}:0] INJECTED_WEIGHTS = "
            f"{total_bits_w}'h{Weights:0{hex_width}x};\n"
        )

    total_bits_b = int(OutChannels) * int(BiasBits)
    vhb_path = os.path.join(custom_work_dir, "injected_biases.vh")
    with open(vhb_path, "w") as f:
        hex_width = (total_bits_b + 3) // 4
        f.write(
            f"localparam logic signed [{total_bits_b-1}:0] INJECTED_BIASES = "
            f"{total_bits_b}'h{Biases:0{hex_width}x};\n"
        )

    # ---- Pass big ints via env vars for cocotb ----
    os.environ["INJECTED_WEIGHTS_INT"] = str(Weights)
    os.environ["INJECTED_BIASES_INT"]  = str(Biases)

    wrapper_path = os.path.join(tbpath, "tb_linear_layer.sv")

    runner(
        simulator=simulator,
        timescale=timescale,
        tbpath=tbpath,
        params=parameters,
        testname=test_name,
        work_dir=custom_work_dir,
        includes=[custom_work_dir],        # so injected_*.vh can be `included
        toplevel_override="tb_linear_layer",
        extra_sources=[wrapper_path],
    )

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, InChannels, OutBits, OutChannels, BiasBits", 
                         [(1, 2, 1, output_width(1, 2, 1), 1, 2)])
def test_lint(simulator, InBits, WeightBits, InChannels, OutBits, OutChannels, BiasBits):
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, InChannels, OutBits, OutChannels, BiasBits", 
                         [(1, 2, 1, output_width(1, 2, 1), 1, 2)])
def test_style(simulator, InBits, WeightBits, InChannels, OutBits, OutChannels, BiasBits):
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
    """Drive exactly one input vector (one handshake) and expect exactly one output vector."""

    IC = int(dut.InChannels.value)
    OC = int(dut.OutChannels.value)
    WW = int(dut.WeightBits.value)
    BW = int(dut.BiasBits.value)

    # One accepted input -> one produced output (LINEAR vector-per-handshake assumption)
    N_in  = 1
    N_out = 1
    rate  = 1.0

    # ---- Unpack injected weights (and biases) ----
    weights_2d = unpack_weights(int(os.environ["INJECTED_WEIGHTS_INT"]), WW, OC, IC)
    biases_1d = unpack_biases(int(os.environ["INJECTED_BIASES_INT"]), BW, OC)
    
    # Instantiate PyTorch reference model
    linear = torch_linear_ref(weights_2d, biases_1d, IC, OC)
    linear.eval()

    model = LinearLayerModel(dut, weights_2d, biases_1d)
    m = ModelRunner(dut, model)

    om = OutputModel(dut, RateGenerator(dut, 1.0), N_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_in)

    # ---- Init drives ----
    dut.ready_i.value = 0
    dut.valid_i.value = 0
    dut.data_i.value  = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    # ---- Start coroutines ----
    m.start()
    om.start()
    im.start()

    # ---- Wait for exactly one output handshake ----
    # Generous timeout: a few cycles + reset slack
    tmo_ns = 2000
    timed_out = False
    try:
        await om.wait(tmo_ns)
    except SimTimeoutError:
        timed_out = True
    
    assert not timed_out, (
        f"Timed out waiting for the single LINEAR output handshake. "
        f"Produced={om.nproduced()} Expected={N_out}"
    )

    # ---- Stop driving ----
    dut.valid_i.value = 0
    dut.ready_i.value = 0

async def rate_tests(dut, in_rate: float, out_rate: float, N_vec: int = 200):
    """
    LINEAR fuzz test: drive N_vec input vectors; expect N_vec output vectors.
    Assumes ONE handshake on input == one full vector (packed [IC] samples),
    and ONE handshake on output == one full output vector (packed [OC]).
    """

    IC = int(dut.InChannels.value)
    OC = int(dut.OutChannels.value)
    WW = int(dut.WeightBits.value)
    BW = int(dut.BiasBits.value)

    N_in  = int(N_vec)
    N_out = int(N_vec)

    # --- Unpack injected weights ---
    weights_2d = unpack_weights(int(os.environ["INJECTED_WEIGHTS_INT"]), WW, OC, IC)
    biases_1d  = unpack_biases(int(os.environ["INJECTED_BIASES_INT"]), BW, OC)

    # Instantiate PyTorch reference model
    linear = torch_linear_ref(weights_2d, biases_1d, IC, OC)
    linear.eval()

    # --- Model + runner ---
    model = LinearLayerModel(dut, weights_2d, biases_1d, linear)
    m = ModelRunner(dut, model)

    # --- Producer/consumer with fuzzed rates ---
    om = OutputModel(dut, RateGenerator(dut, out_rate), N_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, in_rate), N_in)

    # --- Reset/bringup ---
    dut.ready_i.value = 0
    dut.valid_i.value = 0
    dut.data_i.value  = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # --- Timeout sizing: scale by bottleneck rate ---
    slow = min(max(in_rate, 1e-3), max(out_rate, 1e-3))
    slow = max(min(slow, 1.0), 0.02)

    timeout_ns = int((N_out * 50 + 500) / slow)

    try:
        await om.wait(timeout_ns)

    except SimTimeoutError:
        assert 0, (
            f"Timed out in LINEAR rate test. "
            f"Expected {N_out} output handshakes, got {om.nproduced()} "
            f"in {timeout_ns} ns (in_rate={in_rate}, out_rate={out_rate})."
        )

@cocotb.test
async def out_fuzz_test(dut):
    await rate_tests(dut, in_rate=1.0, out_rate=0.5, N_vec=100)

@cocotb.test
async def in_fuzz_test(dut):
    await rate_tests(dut, in_rate=0.5, out_rate=1.0, N_vec=100)

@cocotb.test
async def inout_fuzz_test(dut):
    await rate_tests(dut, in_rate=0.5, out_rate=0.5, N_vec=100)

@cocotb.test
async def full_bw_test(dut):
    await rate_tests(dut, in_rate=1.0, out_rate=1.0, N_vec=100)