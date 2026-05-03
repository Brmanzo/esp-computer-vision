# test_classifier_layer.py
import numpy as np
import os
from   pathlib import Path
import pytest
import shutil
import torch
from   torch  import nn


from util.utilities  import runner, lint, clock_start_sequence, reset_sequence, \
                            auto_unpack, load_tests_from_csv, inject_weights_and_biases
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

gen_rules = [
    ("Weights", lambda WeightBits, ClassCount, InChannels: gen_weights(WeightBits, ClassCount, InChannels, seed=1234)),
    ("Biases",  lambda BiasBits, ClassCount: gen_biases(BiasBits, ClassCount, seed=1234)),
]

TEST_CASES = load_tests_from_csv(os.path.join(tbpath, "test_cases.csv"), gen_rules=gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)
def test_each(test_name, simulator,
              TermBits, TermCount, BusBits, InChannels,
              ClassCount, WeightBits, Weights, BiasBits, Biases):
    parameters = dict(locals())

    # Remove injected params so cocotb-runner doesn't pass them on CLI
    parameters.pop("Weights", None)
    parameters.pop("Biases", None)

    param_str = f"TermBits_{TermBits}_WeightBits_{WeightBits}_BiasBits_{BiasBits}_test_{test_name}"
    
    weight_bits = ClassCount * InChannels * WeightBits
    bias_bits   = ClassCount * BiasBits

    custom_work_dir = inject_weights_and_biases(
        simulator=simulator, parameters=parameters, param_str=param_str, 
        tbpath=tbpath, test_class="each", Weights=Weights, Biases=Biases, 
        weight_bits=weight_bits, bias_bits=bias_bits, layer=0)
        
    wrapper_path = os.path.join(tbpath, "tb_classifier_layer.sv")
    runner(
        simulator=simulator, timescale=timescale, tbpath=tbpath, params=parameters,
        pymodule="test_classifier_layer", testname=test_name, work_dir=custom_work_dir,
        sim_build=custom_work_dir, includes=[custom_work_dir], toplevel_override="tb_classifier_layer", 
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

    weights_2d = unpack_weights(int(os.environ["INJECTED_WEIGHTS_0_INT"]), WW, OC, IC)
    biases_1d = unpack_biases(int(os.environ["INJECTED_BIASES_0_INT"]), BW, OC)

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

    weights_2d = unpack_weights(int(os.environ["INJECTED_WEIGHTS_0_INT"]), WW, OC, IC)
    biases_1d = unpack_biases(int(os.environ["INJECTED_BIASES_0_INT"]), BW, OC)

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
