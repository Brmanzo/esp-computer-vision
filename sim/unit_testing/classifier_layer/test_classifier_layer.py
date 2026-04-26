# test_classifier_layer.py
import numpy as np
import os
from   pathlib import Path
import pytest
import shutil
import torch
from   torch  import nn


from util.utilities  import runner, lint, clock_start_sequence, reset_sequence
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from util.bitwise    import unpack_weights, unpack_biases
from util.gen_inputs import gen_weights, gen_biases
from functional_models.classifier_layer import ClassifierLayerModel, RandomDataGenerator

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

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("TermBits, TermCount, BusBits, InChannels, ClassCount, WeightBits, Weights, Biases, BiasBits", [
    (1,  32, 8,  8, 10,  2, gen_weights( 8, 10, 2), gen_biases( 2, 10), 2),
    (2,  64, 8, 16, 10,  4, gen_weights(16, 10, 4), gen_biases( 4, 10), 4),
    (4, 128, 8, 24, 10,  5, gen_weights(24, 10, 5), gen_biases( 8, 10), 8),
    (8, 256, 8, 32, 10,  8, gen_weights(32, 10,	8), gen_biases(16, 10),	16)
])
def test_each(test_name, simulator, TermBits, TermCount, BusBits, InChannels, ClassCount, WeightBits, Weights, BiasBits, Biases):
    # This line must be first
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']

    # Remove injected params so cocotb-runner doesn't pass them on CLI
    del parameters["Weights"]
    del parameters["Biases"]

    param_str = f"TermBits_{TermBits}_WeightBits_{WeightBits}_BiasBits_{BiasBits}_test_{test_name}"
    custom_work_dir = os.path.join(tbpath, "run", "width", param_str, simulator)
    if simulator.startswith("icarus") and os.path.exists(custom_work_dir):
        shutil.rmtree(custom_work_dir)
    os.makedirs(custom_work_dir, exist_ok=True)

    # ---- Emit injected_weights.vh ----
    total_bits_w = int(ClassCount) * int(InChannels) * int(WeightBits)
    vh_path = os.path.join(custom_work_dir, "injected_weights.vh")
    with open(vh_path, "w") as f:
        hex_width = (total_bits_w + 3) // 4
        f.write(
            f"localparam logic signed [{total_bits_w-1}:0] INJECTED_WEIGHTS = "
            f"{total_bits_w}'h{Weights:0{hex_width}x};\n"
        )

    total_bits_b = int(ClassCount) * int(BiasBits)
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

    wrapper_path = os.path.join(tbpath, "tb_classifier_layer.sv")
    runner(
        simulator=simulator,
        timescale=timescale,
        tbpath=tbpath,
        params=parameters,
        testname=test_name,
        work_dir=custom_work_dir,
        includes=[custom_work_dir],        # so injected_*.vh can be `included
        toplevel_override="tb_classifier_layer",
        extra_sources=[wrapper_path],
    )

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("TermBits, TermCount, BusBits, InChannels, ClassCount, WeightBits, Weights, BiasBits, Biases", [
    (2, 10, 8, 2, 4, 2, gen_weights(2, 4, 2), 2, gen_biases(2, 10))
])
def test_lint(simulator, TermBits, TermCount, BusBits, InChannels, ClassCount, WeightBits, Weights, BiasBits, Biases):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    del parameters["Weights"]
    del parameters["Biases"]
    lint(simulator, timescale, tbpath, parameters, pymodule="test_classifier_layer")

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("TermBits, TermCount, BusBits, InChannels, ClassCount, WeightBits, Weights, BiasBits, Biases", [
    (2, 10, 8, 2, 4, 2, gen_weights(2, 4, 2), 2, gen_biases(2, 10))
])
def test_style(simulator, TermBits, TermCount, BusBits, InChannels, ClassCount, WeightBits, Weights, BiasBits, Biases):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    del parameters["Weights"]
    del parameters["Biases"]
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"], pymodule="test_classifier_layer")

@cocotb.test
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    await clock_start_sequence(clk_i)
    await reset_sequence(clk_i, rst_i, 10)

@cocotb.test
async def single_test(dut):
    T = int(dut.TermCount.value)

    IC = int(dut.InChannels.value)
    OC = int(dut.ClassCount.value)
    WW = int(dut.WeightBits.value)
    BW = int(dut.BiasBits.value)

    weights_2d = unpack_weights(int(os.environ["INJECTED_WEIGHTS_INT"]), WW, OC, IC)
    biases_1d = unpack_biases(int(os.environ["INJECTED_BIASES_INT"]), BW, OC)

    model = ClassifierLayerModel(dut, weights_2d, biases_1d)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, 1), 1)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, 1), T)

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    timeout_ns = 300
    await om.wait(timeout_ns)

async def rate_tests(dut, in_rate, out_rate):
    """Input random data elements at 100% line rate"""

    eg = RandomDataGenerator(dut)
    TC = int(dut.TermCount.value)
    groups = 10
    l_in = groups * TC  
    l_out = groups

    IC = int(dut.InChannels.value)
    OC = int(dut.ClassCount.value)
    WW = int(dut.WeightBits.value)
    BW = int(dut.BiasBits.value)

    weights_2d = unpack_weights(int(os.environ["INJECTED_WEIGHTS_INT"]), WW, OC, IC)
    biases_1d = unpack_biases(int(os.environ["INJECTED_BIASES_INT"]), BW, OC)

    model = ClassifierLayerModel(dut, weights_2d, biases_1d)
    m = ModelRunner(dut, model)
    om = OutputModel(dut, RateGenerator(dut, out_rate), l_out)
    im = InputModel(dut, eg, RateGenerator(dut, in_rate), l_in)

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

    clock_period_ns = 10
    slow = min(in_rate, out_rate)
    slow = max(slow, 0.05) 
    timeout_ns = int(((l_in + 500) / slow) * clock_period_ns)

    try:
        await om.wait(timeout_ns)
    except SimTimeoutError:
        assert 0, (
            f"Test timed out. Expected {l_out} outputs from {l_in} inputs "
            f"with TermCount={TC}, in_rate={in_rate}, out_rate={out_rate}"
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
