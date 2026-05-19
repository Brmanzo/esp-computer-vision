# test_neuron.py
import math
import numpy as np
import os
from   pathlib import Path
import pytest
import queue
import torch
from   typing import List

from cocotb.clock import Clock
from util.utilities import runner, lint, assert_resolvable, auto_unpack, \
                           sim_verbose, inject_weights_and_biases, load_tests_from_csv, \
                           clock_start_sequence, reset_sequence
from util.bitwise   import sign_extend, pack_terms, unpack_terms, unpack_weights
from util.torch_ref import torch_neuron_ref
from util.gen_inputs import gen_weights, gen_biases, gen_input_channels
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import Decimal, Timer, FallingEdge, RisingEdge
   
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

auto_rules = [
     ("OutBits", "OutBits", lambda InBits, WeightBits, InChannels, BiasBits: output_width(InBits, WeightBits, InChannels, BiasBits)),
]

gen_rules = [
    ("Weights", lambda WeightBits, InChannels: gen_weights(WeightBits, InChannels, seed=1234)),
    ("Biases",  lambda BiasBits: gen_biases(BiasBits, seed=1234))
]
TEST_CASES_WIDTH = load_tests_from_csv(os.path.join(tbpath, "test_cases_width.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES_WIDTH)
def test_width(test_name, simulator, use_dsp,
               InBits, WeightBits, InChannels, OutBits, BiasBits, Weights, Biases):
    if simulator == "icarus" and use_dsp:
        pytest.skip("Icarus Verilog has issues with string parameters for ROM initialization")
    if use_dsp:
        # neuron_dsp has a fixed 32-bit accumulator.
        # Calculate required bits: max(OutBits, WeightBits + InBits + log2(InChannels))
        import math
        req_bits = max(int(OutBits), int(WeightBits) + int(InBits) + math.ceil(math.log2(int(InChannels))))
        if req_bits > 32:
            pytest.skip(f"neuron_dsp accumulator (32 bits) too small for required {req_bits} bits")

    parameters = {
        "InBits":     InBits,
        "WeightBits": WeightBits,
        "InChannels": InChannels,
        "OutBits":    OutBits,
        "BiasBits":   BiasBits,
        "GEN_DSP":    1 if use_dsp else 0
    }

    param_str = f"InBits_{InBits}_WeightBits_{WeightBits}_OutBits_{OutBits}_BiasBits_{BiasBits}_test_{test_name}"
    weight_bits  = int(WeightBits)
    bias_bits    = int(BiasBits)
    weight_count = int(InChannels)

    custom_work_dir = inject_weights_and_biases(
        simulator=simulator, parameters=parameters, param_str=param_str,
        tbpath=tbpath, test_class="width", Weights=Weights, Biases=Biases,
        weight_bits=weight_bits, bias_bits=bias_bits, weight_count=weight_count,
        layer=0, dsp_count=int(use_dsp))

    wrapper_path = os.path.join(tbpath, "tb_neuron.sv")
    # Repo root is 3 levels up from sim/unit_testing/neuron
    repo_root = Path(__file__).parent.parent.parent.parent
    jsonname = "neuron_dsp.json" if use_dsp else "neuron.json"
    
    runner(
        simulator=simulator, timescale=timescale, tbpath=tbpath, params=parameters,
        testname=test_name, work_dir=custom_work_dir, includes=[custom_work_dir],
        toplevel_override="tb_neuron", extra_sources=[wrapper_path],
        jsonname=jsonname, pymodule="test_neuron"
    )

TEST_CASES_CHANNELS = load_tests_from_csv(os.path.join(tbpath, "test_cases_channels.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES_CHANNELS)
def test_channels(test_name, simulator, use_dsp,
                  InBits, WeightBits, InChannels, OutBits, BiasBits, Weights, Biases):
    if simulator == "icarus" and use_dsp:
        pytest.skip("Icarus Verilog has issues with string parameters for ROM initialization")
    # Icarus Verilog has a known bug with signed arithmetic on packed arrays wider than
    # 128 bits. Skip Icarus for configurations that exceed this threshold.
    if simulator == "icarus" and (int(InChannels) * int(WeightBits) > 128):
        pytest.skip(f"Icarus unreliable for packed bus width {int(InChannels) * int(WeightBits)} > 128 bits")

    if use_dsp:
        import math
        req_bits = max(int(OutBits), int(WeightBits) + int(InBits) + math.ceil(math.log2(int(InChannels))))
        if req_bits > 32:
            pytest.skip(f"neuron_dsp accumulator (32 bits) too small for required {req_bits} bits")

    parameters = {
        "InBits":     InBits,
        "WeightBits": WeightBits,
        "InChannels": InChannels,
        "OutBits":    OutBits,
        "BiasBits":   BiasBits,
        "GEN_DSP":    1 if use_dsp else 0
    }

    param_str = f"InChannels_{InChannels}_test_{test_name}"
    weight_bits  = int(WeightBits)
    bias_bits    = int(BiasBits)
    weight_count = int(InChannels)

    custom_work_dir = inject_weights_and_biases(
        simulator=simulator, parameters=parameters, param_str=param_str,
        tbpath=tbpath, test_class="channels", Weights=Weights, Biases=Biases,
        weight_bits=weight_bits, bias_bits=bias_bits, weight_count=weight_count,
        layer=0, dsp_count=int(use_dsp))

    wrapper_path = os.path.join(tbpath, "tb_neuron.sv")
    # Repo root is 3 levels up from sim/unit_testing/neuron
    repo_root = Path(__file__).parent.parent.parent.parent
    jsonname = "neuron_dsp.json" if use_dsp else "neuron.json"

    runner(
        simulator=simulator, timescale=timescale, tbpath=tbpath, params=parameters,
        testname=test_name, work_dir=custom_work_dir, includes=[custom_work_dir],
        toplevel_override="tb_neuron", extra_sources=[wrapper_path],
        jsonname=jsonname, pymodule="test_neuron"
    )

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, InChannels, OutBits", 
                         [(1, 2, 1, output_width(1, 2, 1))])
def test_lint(simulator, InBits, WeightBits, InChannels, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, jsonname="neuron.json")

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("InBits, WeightBits, InChannels, OutBits, Weights, BiasBits, Bias",
                         [(1, 2, 1, output_width(1, 2, 1), gen_weights(2, 1, seed=1234), 2, gen_biases(2))])
def test_style(simulator, InBits, WeightBits, InChannels, OutBits, Weights, BiasBits, Bias):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"], jsonname="neuron.json")

class NeuronModel():
    def __init__(self, dut, weights: List[int], bias: int):
        self._dut = dut
        self._data_o = dut.data_o
        self._data_i = dut.data_i

        self._q: 'queue.SimpleQueue[List[int]]' = queue.SimpleQueue() # Buffers real input while processing padding

        self._InBits     = int(dut.InBits.value)
        self._OutBits    = int(dut.OutBits.value)
        self._InChannels  = int(dut.InChannels.value)
        self._WeightBits = int(dut.WeightBits.value)
        self._BiasBits   = int(dut.BiasBits.value)

        self.w = np.array(weights, dtype=int)

        bias_bits = bias & ((1 << self._BiasBits) - 1)
        self.b = sign_extend(bias_bits, self._BiasBits)

        self._in_idx = 0

    def consume(self):
        packed = int(self._data_i.value.integer)
        
        din = unpack_terms(packed, int(self._dut.InBits.value), self._InChannels)
        
        # compute expected NOW, while _buf matches this accepted input position    
        acc = self.b
        
        for ic in range(self._InChannels):
            # 1. Encode binary activation from {0,1} to {-1,1}
            if self._InBits == 1:
                val = 1 if din[ic] == 1 else -1
            else:
                val = int(din[ic])
                
            # 2. Accumulate
            acc += int(self.w[ic]) * val

        # 3. No Activation Function (Raw output)
        return acc

    def produce(self, expected, ref=None):
        # Use the top-level wide accumulator for verification
        if hasattr(self._dut, "acc_full"):
            sig = self._dut.acc_full
        else:
            sig = self._data_o

        assert_resolvable(sig)
        
        gen_dsp = hasattr(self._dut, "GEN_DSP") and int(self._dut.GEN_DSP.value) == 1
        
        if gen_dsp:
            # In sequential DSP mode, accumulation happens in OutBits width
            got = sign_extend(int(sig.value.integer), self._OutBits)
            exp = sign_extend(int(expected) & ((1 << self._OutBits) - 1), self._OutBits)
        else:
            got = sign_extend(int(sig.value.integer), 32 if sig == getattr(self._dut, "acc_full", None) else self._OutBits)
            exp = int(expected)

        assert got == exp, (
            f"Mismatch. Expected {exp}, got {got}, Weights: {self.w}, Bias: {self.b}"
        )
        
        actual_ref = ref if ref is not None else getattr(self, "ref", None)
        
        if gen_dsp and actual_ref is not None:
            ref_int = int(round(actual_ref))
            actual_ref = float(sign_extend(ref_int & ((1 << self._OutBits) - 1), self._OutBits))
            
        if sim_verbose():
            print(f"got: {got}, expected: {exp}, Pytorch Ref: {actual_ref}")
        assert actual_ref is None or math.isclose(got, actual_ref, rel_tol=1e-5), (
            f"Mismatch with Pytorch. Expected {actual_ref}, got {got}, Weights: {self.w}, Bias: {self.b}"
        )
    
async def comb_step(dut, model, din_list, ref):
    if hasattr(dut, "GEN_DSP") and int(dut.GEN_DSP.value) == 1:
        from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
        
        model.ref = ref
        m = ModelRunner(dut, model)

        class SingleDataGenerator:
            def __init__(self, packed, raw):
                self.packed = packed
                self.raw = raw
            def generate(self):
                return self.packed, self.raw

        packed = pack_terms(din_list, int(dut.InBits.value))
        im = InputModel(dut, SingleDataGenerator(packed, din_list), RateGenerator(dut, 1.0), 1)
        om = OutputModel(dut, RateGenerator(dut, 1.0), 1)

        m.start()
        im.start()
        om.start()

        # Calculate appropriate sequential timeout based on InChannels
        IC = int(dut.InChannels.value)
        cycles_per_vec = IC + 10
        timeout_ns = int(cycles_per_vec * 10 + 2000)

        from cocotb.result import SimTimeoutError
        try:
            await om.wait(timeout_ns)
        except SimTimeoutError:
            assert False, f"Timed out waiting for sequential neuron handshake"

        m.stop()
        im.stop()
        om.stop()
    else:
        # Combinatorial: drive packed input and check after a delta
        dut.data_i.value = pack_terms(din_list, int(dut.InBits.value))
        await Timer(Decimal(1), units="step")
        expected = model.consume()
        model.produce(expected, ref)

@cocotb.test
async def single_test(dut):
    IC = int(dut.InChannels.value)
    WW = int(dut.WeightBits.value)
    in_bits = int(dut.InBits.value)

    # Start clock and reset if using DSP sequential module
    gen_dsp = hasattr(dut, "GEN_DSP") and int(dut.GEN_DSP.value) == 1
    if gen_dsp:
        dut.valid_i.value = 0
        dut.ready_i.value = 0
        await clock_start_sequence(dut.clk_i, period=10, unit="ns")
        await reset_sequence(dut.clk_i, dut.rst_i, cycles=2)

    dut.data_i.value = 0
    await Timer(Decimal(1), units="step")

    weights_2d = unpack_weights(
        packed_val=int(os.environ["INJECTED_WEIGHTS_0_INT"], 0),
        WW=WW,
        OC=1,
        IC=IC
    )
    weights_1d = weights_2d[0]

    bias_raw = int(os.environ["INJECTED_BIASES_0_INT"], 0)
    bias = sign_extend(bias_raw, int(dut.BiasBits.value))

    fc = torch_neuron_ref(weights_1d, bias, IC)

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

    # Start clock and reset if using DSP sequential module
    gen_dsp = hasattr(dut, "GEN_DSP") and int(dut.GEN_DSP.value) == 1
    if gen_dsp:
        dut.valid_i.value = 0
        dut.ready_i.value = 0
        await clock_start_sequence(dut.clk_i, period=10, unit="ns")
        await reset_sequence(dut.clk_i, dut.rst_i, cycles=2)

    dut.data_i.value = 0
    await Timer(Decimal(1), units="step")

    weights_2d = unpack_weights(
        packed_val=int(os.environ["INJECTED_WEIGHTS_0_INT"], 0),
        WW=WW,
        OC=1,
        IC=IC
    )
    weights_1d = weights_2d[0]
    # -----------------------------------------------------------

    bias_raw = int(os.environ["INJECTED_BIASES_0_INT"], 0)
    bias = sign_extend(bias_raw, BW)

    # Instantiate PyTorch Reference Model using the signed bias
    fc = torch_neuron_ref(weights_1d, bias, IC)
    model = NeuronModel(dut, weights_1d, bias)

    for _ in range(500):
        # 1. Generate DUT-encoded random inputs
        din = gen_input_channels(in_bits, IC, seed=1234)

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