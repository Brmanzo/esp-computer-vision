import os
import sys
import git
import queue
import math
import numpy as np
from typing import List
from decimal import Decimal
import torch
import torch.nn as nn
import shutil

_REPO_ROOT = git.Repo(search_parent_directories=True).working_tree_dir
assert _REPO_ROOT is not None, "REPO_ROOT path must not be None"
assert (os.path.exists(_REPO_ROOT)), "REPO_ROOT path must exist"
_UTIL_PATH = os.path.join(_REPO_ROOT, "sim", "util")
assert os.path.exists(_UTIL_PATH), f"Utilities path does not exist: {_UTIL_PATH}"
sys.path.insert(0, _UTIL_PATH)
from utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence, delay_cycles
tbpath = os.path.dirname(os.path.realpath(__file__))

import pytest

import cocotb

from cocotb.triggers import Timer

from cocotb_test.simulator import run
   
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

def sign_extend(val: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (val ^ sign) - sign

def gen_weights(WW: int, IC: int, seed: int | None = None):
    rng = random.Random(seed)
    if WW < 2:
        raise ValueError("Weight width must be at least 2 to include negative values in test kernels.")
    elif WW == 2:
        rand_weight_value = lambda: rng.choice([-1, 0, 1])
    else:
        max_val = (1 << (WW - 1)) - 1
        min_val = -(1 << (WW - 1))
        rand_weight_value = lambda: rng.randint(min_val, max_val)

    # Generate the 4D matrix
    weights1 = [rand_weight_value() for _ in range(IC)]

    # Pack the 2D weights into a single integer for the Verilog Parameter
    packed_weights = 0
    bit_shift = 0
    mask = (1 << WW) - 1

    # Iterate from LSB to MSB: cols -> rows -> InChannels -> OutChannels
    for ic in range(IC):
        w = weights1[ic]

        w_bits = w if w >= 0 else (1 << WW) + w
        w_bits = w_bits & mask
        
        # Shift and combine into the main bit vector
        packed_weights |= (w_bits << bit_shift)
        bit_shift += WW

    return packed_weights

def gen_bias(BW: int, seed: int | None = None):
    rng = random.Random(seed)

    max_val = (1 << (BW - 1)) - 1
    min_val = -(1 << (BW - 1))
    rand_bias_value = lambda: rng.randint(min_val, max_val)

    return rand_bias_value()

def unpack_weights(packed_val: int, WW: int, IC: int):
    """Reconstructs the 4D weights matrix from the Verilog parameter integer."""
    mask = (1 << WW) - 1
    sign_bit = 1 << (WW - 1)
    
    weights1 = []
    bit_shift = 0
    
    # Must mirror the exact same LSB -> MSB iteration order used in packing

    for _ in range(IC):
        # Extract the specific bits for this weight
        w_bits = (packed_val >> bit_shift) & mask
        
        # Convert from two's complement back to a signed Python integer
        if w_bits & sign_bit:
            w = w_bits - (1 << WW)
        else:
            w = w_bits
            
        weights1.append(w)
        bit_shift += WW
        
    return weights1

def pack_data_i(samples, width):
    """
    samples: list[int] length = InChannels
    width: WidthIn
    Returns packed int: sum(samples[ic] << (ic*width))
    """
    mask = (1 << width) - 1
    out = 0
    for ic, v in enumerate(samples):
        out |= (int(v) & mask) << (ic * width)
    return out

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
def unpack_data_i(packed, width_in, IC):
    mask = (1 << width_in) - 1
    return [ (packed >> (ic * width_in)) & mask for ic in range(IC) ]

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize(
    "WidthIn, WeightWidth, InChannels, WidthOut, Weights, BiasWidth, Bias",
    [
        (1, 2, 1, output_width(1, 2, 1), gen_weights(2, 1, seed=1234), 2, gen_bias(2)),
        (2, 3, 1, output_width(2, 3, 1), gen_weights(3, 1, seed=1234), 3, gen_bias(3)),
        (4, 5, 1, output_width(4, 5, 1), gen_weights(5, 1, seed=1234), 5, gen_bias(5)),
        (8, 8, 1, output_width(8, 8, 1), gen_weights(8, 1, seed=1234), 8, gen_bias(8)),
    ],
)
def test_width(test_name, simulator, WidthIn, WeightWidth, InChannels, WidthOut, Weights, BiasWidth, Bias):
    parameters = dict(locals())
    del parameters["test_name"]
    del parameters["simulator"]

    # Remove injected params so cocotb-runner doesn't pass them on CLI
    del parameters["Weights"]
    del parameters["Bias"]

    param_str = f"WidthIn_{WidthIn}_WeightWidth_{WeightWidth}_WidthOut_{WidthOut}_test_{test_name}"
    custom_work_dir = os.path.join(tbpath, "run", "width", param_str, simulator)
    if simulator.startswith("icarus") and os.path.exists(custom_work_dir):
        shutil.rmtree(custom_work_dir)
    os.makedirs(custom_work_dir, exist_ok=True)

    # ---- Emit injected_weights.vh ----
    total_bits_w = int(InChannels) * int(WeightWidth)
    vh_path = os.path.join(custom_work_dir, "injected_weights.vh")
    with open(vh_path, "w") as f:
        hex_width = (total_bits_w + 3) // 4
        weights_bits = Weights & ((1 << total_bits_w) - 1)
        f.write(
            f"localparam logic signed [{total_bits_w-1}:0] INJECTED_WEIGHTS = "
            f"{total_bits_w}'h{weights_bits:0{hex_width}x};\n"
        )
    # ---- Emit injected_bias.vh ----
    total_bits_b = BiasWidth
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
    "WidthIn, WeightWidth, InChannels, WidthOut, Weights, BiasWidth, Bias",
    [
        (1, 2, 1,  output_width(1, 2, 1),  gen_weights(2, 1, seed=1234),  8, gen_bias(8)),
        (1, 2, 2,  output_width(1, 2, 2),  gen_weights(2, 2, seed=1234),  8, gen_bias(8)),
        (1, 2, 4,  output_width(1, 2, 4),  gen_weights(2, 4, seed=1234),  8, gen_bias(8)),
        (1, 2, 32, output_width(1, 2, 32), gen_weights(2, 32, seed=1234), 8, gen_bias(8)),
    ],
)
def test_channels(test_name, simulator, WidthIn, WeightWidth, InChannels, WidthOut, Weights, BiasWidth, Bias):
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
    total_bits_w = int(InChannels) * int(WeightWidth)
    vh_path = os.path.join(custom_work_dir, "injected_weights.vh")
    with open(vh_path, "w") as f:
        hex_width = (total_bits_w + 3) // 4
        weights_bits = Weights & ((1 << total_bits_w) - 1)
        f.write(
            f"localparam signed [{total_bits_w-1}:0] INJECTED_WEIGHTS = "
            f"{total_bits_w}'h{weights_bits:0{hex_width}x};\n"
        )
    # ---- Emit injected_bias.vh ----
    total_bits_b = BiasWidth
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
@pytest.mark.parametrize("WidthIn, WeightWidth, InChannels, WidthOut", 
                         [(1, 2, 1, output_width(1, 2, 1))])
def test_lint(simulator, WidthIn, WeightWidth, InChannels, WidthOut):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("WidthIn, WeightWidth, InChannels, WidthOut, Weights, BiasWidth, Bias", 
                         [(1, 2, 1, output_width(1, 2, 1), gen_weights(2, 1, seed=1234), 2, gen_bias(2))])
def test_style(simulator, WidthIn, WeightWidth, InChannels, WidthOut, Weights, BiasWidth, Bias):
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

        self._WidthIn     = int(dut.WidthIn.value)
        self._WidthOut    = int(dut.WidthOut.value)
        self._InChannels  = int(dut.InChannels.value)
        self._WeightWidth = int(dut.WeightWidth.value)
        self._BiasWidth   = int(dut.BiasWidth.value)

        self.w = np.array(weights, dtype=int)

        bias_bits = int(bias) & ((1 << self._BiasWidth) - 1)
        self.b = sign_extend(bias_bits, self._BiasWidth)

        self._in_idx = 0

    def consume(self):
        packed = int(self._data_i.value.integer)
        din = unpack_data_i(packed, int(self._dut.WidthIn.value), self._InChannels)
        # compute expected NOW, while _buf matches this accepted input position    
        acc = self.b
        for ic in range(self._InChannels):
            acc += int(self.w[ic]) * int(din[ic])

        return acc

    def produce(self, expected, ref=None):
        assert_resolvable(self._data_o)

        w   = int(self._dut.WidthOut.value)
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
    w = int(dut.WidthIn.value)
    dut.data_i.value = pack_data_i(din_list, w)

    # Compute expected from the *driven input* (your existing consume reads dut.data_i)
    await Timer(Decimal(1), units="step")
    expected = model.consume()

    # Check output using your existing produce()
    model.produce(expected, ref)

@cocotb.test
async def single_test(dut):
    IC = int(dut.InChannels.value)
    WW = int(dut.WeightWidth.value)
    dut.data_i.value = 0  # Initialize to avoid X's in consume()
    await Timer(Decimal(1), units="step")

    weights_1d = unpack_weights(int(os.environ["INJECTED_WEIGHTS_INT"]), WW, IC)
    bias = int(os.environ["INJECTED_BIAS_INT"])

    # Instantiate PyTorch Reference Model
    fc = fc_reference(weights_1d, bias, IC)
    print(f"Testing with Weights: {weights_1d}, Bias: {bias}")
    model = NeuronModel(dut, weights_1d, bias)

    din = [random.randint(0, (1 << int(dut.WidthIn.value)) - 1) for _ in range(IC)]
    # Convert data_i to tensor and compute Pytorch Reference Activation
    tensor = torch.tensor([din], dtype=torch.float32)
    ref = fc(tensor).item()
    await comb_step(dut, model, din, ref)

@cocotb.test
async def full_bw_test(dut):
    IC = int(dut.InChannels.value)
    WW = int(dut.WeightWidth.value)
    dut.data_i.value = 0  # Initialize to avoid X's in consume()
    await Timer(Decimal(1), units="step")

    weights_1d = unpack_weights(int(os.environ["INJECTED_WEIGHTS_INT"]), WW, IC)
    bias = int(os.environ["INJECTED_BIAS_INT"])

    # Instantiate PyTorch Reference Model
    fc = fc_reference(weights_1d, bias, IC)
    model = NeuronModel(dut, weights_1d, bias)

    for _ in range(500):
        din = [random.randint(0, (1 << int(dut.WidthIn.value)) - 1) for _ in range(IC)]
        # Convert data_i to tensor and compute Pytorch Reference Activation
        tensor = torch.tensor([din], dtype=torch.float32)
        ref = fc(tensor).item()
        await comb_step(dut, model, din, ref)