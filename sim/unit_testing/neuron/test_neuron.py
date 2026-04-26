# test_neuron.py
from   decimal import Decimal
import math
import numpy as np
import os
from   pathlib import Path
import pytest
import queue
import shutil
import torch
import torch.nn as nn
from   typing import List

from util.utilities import runner, lint, assert_resolvable
from util.bitwise   import sign_extend, pack_terms, unpack_terms, unpack_weights
from util.gen_inputs import gen_weights, gen_biases, gen_input_channels
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import Timer
   
import random
random.seed(50)

timescale = "1ps/1ps"

tests = ['single_test'
        ,'full_bw_test']

def output_width(width_in: int, weight_width: int, in_channels: int=1, bias_width: int=8) -> str:
    '''Calculates proper output width for given input width amount of accumulations.'''
    terms = in_channels

    max_val    = (1 << width_in) - 1       # Unsigned
    max_weight = (1 << (weight_width - 1))
    max_bias   = (1 << (bias_width - 1))

    max_sum = terms * max_val * max_weight + max_bias
    abs_bits = max_sum.bit_length()
    return str(abs_bits + 1)   # +1 for sign bit

def twos_complement(val: int, bits: int) -> int:
    mask = (1 << bits) - 1
    return val & mask

def fc_reference(weights_1d, bias, InChannels):
    # One output channel instantiates a single neuron
    fc = nn.Linear(in_features=InChannels, out_features=1, bias=True)

    # Disable gradient tracking
    fc.weight.requires_grad = False
    fc.bias.requires_grad = False

    # Convert to float32 for deterministic conv math
    fc.weight.data = torch.tensor([weights_1d], dtype=torch.float32)
    fc.bias.data   = torch.tensor([bias], dtype=torch.float32)

    return fc

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize(
    "InBits, WeightBits, InChannels, OutBits, Weights, BiasBits, Bias",
    [
        (1, 2, 1, output_width(1, 2, 1), gen_weights(2, 1, seed=1234), 2, gen_biases(2)),
        (2, 3, 1, output_width(2, 3, 1), gen_weights(3, 1, seed=1234), 3, gen_biases(3)),
        (4, 5, 1, output_width(4, 5, 1), gen_weights(5, 1, seed=1234), 5, gen_biases(5)),
        (8, 8, 1, output_width(8, 8, 1), gen_weights(8, 1, seed=1234), 8, gen_biases(8)),
    ],
)
def test_width(test_name, simulator, InBits, WeightBits, InChannels, OutBits, Weights, BiasBits, Bias):
    parameters = dict(locals())
    del parameters["test_name"]
    del parameters["simulator"]

    # Remove injected params so cocotb-runner doesn't pass them on CLI
    del parameters["Weights"]
    del parameters["Bias"]

    param_str = f"InBits_{InBits}_WeightBits_{WeightBits}_OutBits_{OutBits}_test_{test_name}"
    custom_work_dir = os.path.join(tbpath, "run", "width", param_str, simulator)
    if simulator.startswith("icarus") and os.path.exists(custom_work_dir):
        shutil.rmtree(custom_work_dir)
    os.makedirs(custom_work_dir, exist_ok=True)

    # ---- Emit injected_weights.vh ----
    total_bits_w = int(InChannels) * int(WeightBits)
    vh_path = os.path.join(custom_work_dir, "injected_weights.vh")
    with open(vh_path, "w") as f:
        hex_width = (total_bits_w + 3) // 4
        weights_bits = Weights & ((1 << total_bits_w) - 1)
        f.write(
            f"localparam logic signed [{total_bits_w-1}:0] INJECTED_WEIGHTS = "
            f"{total_bits_w}'h{weights_bits:0{hex_width}x};\n"
        )
    # ---- Emit injected_bias.vh ----
    total_bits_b = BiasBits
    vhb_path = os.path.join(custom_work_dir, "injected_bias.vh")
    with open(vhb_path, "w") as f:
        hex_width = (total_bits_b + 3) // 4
        bias_bits = twos_complement(Bias, total_bits_b)
        f.write(
            f"localparam logic signed [{total_bits_b-1}:0] INJECTED_BIAS = "
            f"{total_bits_b}'h{bias_bits:0{hex_width}x};\n"
        )

    # ---- Pass big ints via env vars for cocotb ----
    os.environ["INJECTED_WEIGHTS_INT"] = str(Weights)
    os.environ["INJECTED_BIAS_INT"]    = str(Bias)

    wrapper_path = os.path.join(tbpath, "tb_neuron.sv")

    runner(
        simulator=simulator,
        timescale=timescale,
        tbpath=tbpath,
        params=parameters,
        testname=test_name,
        work_dir=custom_work_dir,
        includes=[custom_work_dir],        # so injected_*.vh can be `included
        toplevel_override="tb_neuron",
        extra_sources=[wrapper_path],
    )

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize(
    "InBits, WeightBits, InChannels, OutBits, Weights, BiasBits, Bias",
    [
        (1, 2, 1,  output_width(1, 2, 1),  gen_weights(2, 1, seed=1234),  8, gen_biases(8)),
        (1, 2, 2,  output_width(1, 2, 2),  gen_weights(2, 2, seed=1234),  8, gen_biases(8)),
        (1, 2, 4,  output_width(1, 2, 4),  gen_weights(2, 4, seed=1234),  8, gen_biases(8)),
        (1, 2, 32, output_width(1, 2, 32), gen_weights(2, 32, seed=1234), 8, gen_biases(8)),
    ],
)
def test_channels(test_name, simulator, InBits, WeightBits, InChannels, OutBits, Weights, BiasBits, Bias):
    parameters = dict(locals())
    del parameters["test_name"]
    del parameters["simulator"]

    # Remove injected params so cocotb-runner doesn't pass them on CLI
    del parameters["Weights"]
    del parameters["Bias"]

    param_str = f"InChannels_{InChannels}_test_{test_name}"
    custom_work_dir = os.path.join(tbpath, "run", "channels", param_str, simulator)
    if simulator.startswith("icarus") and os.path.exists(custom_work_dir):
        shutil.rmtree(custom_work_dir)
    os.makedirs(custom_work_dir, exist_ok=True)

    # ---- Emit injected_weights.vh ----
    total_bits_w = int(InChannels) * int(WeightBits)
    vh_path = os.path.join(custom_work_dir, "injected_weights.vh")
    with open(vh_path, "w") as f:
        hex_width = (total_bits_w + 3) // 4
        weights_bits = Weights & ((1 << total_bits_w) - 1)
        f.write(
            f"localparam signed [{total_bits_w-1}:0] INJECTED_WEIGHTS = "
            f"{total_bits_w}'h{weights_bits:0{hex_width}x};\n"
        )
    # ---- Emit injected_bias.vh ----
    total_bits_b = BiasBits
    vhb_path = os.path.join(custom_work_dir, "injected_bias.vh")
    with open(vhb_path, "w") as f:
        hex_width = (total_bits_b + 3) // 4
        bias_bits = twos_complement(Bias, total_bits_b)
        f.write(
            f"localparam signed [{total_bits_b-1}:0] INJECTED_BIAS = "
            f"{total_bits_b}'h{bias_bits:0{hex_width}x};\n"
        )

    # ---- Pass big ints via env vars for cocotb ----
    os.environ["INJECTED_WEIGHTS_INT"] = str(Weights)
    os.environ["INJECTED_BIAS_INT"]  = str(Bias)

    wrapper_path = os.path.join(tbpath, "tb_neuron.sv")

    runner(
        simulator=simulator,
        timescale=timescale,
        tbpath=tbpath,
        params=parameters,
        testname=test_name,
        work_dir=custom_work_dir,
        includes=[custom_work_dir],        # so injected_*.vh can be `included
        toplevel_override="tb_neuron",
        extra_sources=[wrapper_path],
    )

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, InChannels, OutBits", 
                         [(1, 2, 1, output_width(1, 2, 1))])
def test_lint(simulator, InBits, WeightBits, InChannels, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, InChannels, OutBits, Weights, BiasBits, Bias", 
                         [(1, 2, 1, output_width(1, 2, 1), gen_weights(2, 1, seed=1234), 2, gen_biases(2))])
def test_style(simulator, InBits, WeightBits, InChannels, OutBits, Weights, BiasBits, Bias):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

class NeuronModel():
    def __init__(self, dut, weights: List[int], bias: int):
        self._dut = dut
        self._data_o = dut.data_o
        self._data_i = dut.data_i

        self._q = queue.SimpleQueue()

        self._InBits     = int(dut.InBits.value)
        self._OutBits    = int(dut.OutBits.value)
        self._InChannels  = int(dut.InChannels.value)
        self._WeightBits = int(dut.WeightBits.value)
        self._BiasBits   = int(dut.BiasBits.value)

        self.w = np.array(weights, dtype=int)

        bias_bits = int(bias) & ((1 << self._BiasBits) - 1)
        self.b = sign_extend(bias_bits, self._BiasBits)

        self._in_idx = 0

    def consume(self):
        packed = int(self._data_i.value.integer)
        
        din = unpack_terms(packed, int(self._dut.InBits.value), self._InChannels)
        
        # compute expected NOW, while _buf matches this accepted input position    
        acc = self.b
        in_bits = int(self._dut.InBits.value)
        
        for ic in range(self._InChannels):
            # 1. Hardware Math Mapping
            if in_bits == 1:
                val = 1 if din[ic] == 1 else -1
            else:
                val = int(din[ic])
                
            # 2. Accumulate
            acc += int(self.w[ic]) * val

        return acc

    def produce(self, expected, ref=None):
        assert_resolvable(self._data_o)

        w   = int(self._dut.OutBits.value)
        got = sign_extend(int(self._data_o.value.integer), w)
        exp = int(expected)

        assert got == exp, (
            f"Mismatch. Expected {exp}, got {got}, Weights: {self.w}, Bias: {self.b}"
        )
        print(f"got: {got}, expected: {exp}, Pytorch Ref: {ref}")
        assert ref is None or math.isclose(got, ref, rel_tol=1e-5), (
            f"Mismatch with Pytorch. Expected {ref}, got {got}, Weights: {self.w}, Bias: {self.b}"
        )
    
async def comb_step(dut, model, din_list, ref):
    # Drive packed input
    w = int(dut.InBits.value)
    dut.data_i.value = pack_terms(din_list, w)

    # Compute expected from the *driven input* (your existing consume reads dut.data_i)
    await Timer(Decimal(1), units="step")
    expected = model.consume()

    # Check output using your existing produce()
    model.produce(expected, ref)

@cocotb.test
async def single_test(dut):
    IC = int(dut.InChannels.value)
    WW = int(dut.WeightBits.value)
    in_bits = int(dut.InBits.value)

    dut.data_i.value = 0
    await Timer(Decimal(1), units="step")

    # --- UPDATED: Pass OC=1 and extract the first dimension ---
    weights_2d = unpack_weights(
        packed_val=int(os.environ["INJECTED_WEIGHTS_INT"]),
        WW=WW,
        OC=1, 
        IC=IC
    )
    weights_1d = weights_2d[0] 
    # -----------------------------------------------------------

    bias_raw = int(os.environ["INJECTED_BIAS_INT"])
    bias = sign_extend(bias_raw, int(dut.BiasBits.value))

    fc = fc_reference(weights_1d, bias, IC)
    print(f"Testing with Weights: {weights_1d}, Bias: {bias}")

    model = NeuronModel(dut, weights_1d, bias)

    # Generate DUT-encoded input values
    din = gen_input_channels(in_bits, IC, seed=1234)

    # Convert only for the PyTorch/math reference
    din_math = [
        1.0 if x == 1 else -1.0
        for x in din
    ] if in_bits == 1 else din

    tensor = torch.tensor([din_math], dtype=torch.float32)
    ref = fc(tensor).item()

    await comb_step(dut, model, din, ref)
    
@cocotb.test
async def full_bw_test(dut):
    IC = int(dut.InChannels.value)
    WW = int(dut.WeightBits.value)
    BW = int(dut.BiasBits.value)
    in_bits = int(dut.InBits.value)

    dut.data_i.value = 0  # Initialize to avoid X's in consume()
    await Timer(Decimal(1), units="step")

    # --- UPDATED: Pass OC=1 and extract the first dimension ---
    weights_2d = unpack_weights(
        packed_val=int(os.environ["INJECTED_WEIGHTS_INT"]),
        WW=WW,
        OC=1,
        IC=IC
    )
    weights_1d = weights_2d[0]
    # -----------------------------------------------------------

    bias_raw = int(os.environ["INJECTED_BIAS_INT"])
    bias = sign_extend(bias_raw, BW)

    # Instantiate PyTorch Reference Model using the signed bias
    fc = fc_reference(weights_1d, bias, IC)
    model = NeuronModel(dut, weights_1d, bias)

    if in_bits == 1:
        min_val, max_val = 0, 1
    else:
        min_val = -(1 << (in_bits - 1))
        max_val =  (1 << (in_bits - 1)) - 1

    for _ in range(500):
        # 1. Generate DUT-encoded random inputs
        din = [random.randint(min_val, max_val) for _ in range(IC)]

        # 2. Convert only for PyTorch/reference math
        din_math = (
            [1.0 if x == 1 else -1.0 for x in din]
            if in_bits == 1
            else din
        )

        # 3. Compute PyTorch reference
        tensor = torch.tensor([din_math], dtype=torch.float32)
        ref = fc(tensor).item()

        await comb_step(dut, model, din, ref)