# test_linear_layer.py
import numpy as np
import os
from   pathlib import Path
import pytest
import shutil
import torch
from   torch import nn
from   typing import List

from util.utilities  import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence, delay_cycles
from util.components import ReadyValidInterface, ModelRunner, RateGenerator, InputModel
from util.bitwise    import sign_extend, pack_terms, unpack_terms
from util.gen_inputs import gen_weights, gen_biases, gen_input_channels
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

def output_width(width_in: int, weight_width: int, in_channels: int=1) -> str:
    '''Calculates proper output width for given input width amount of accumulations.'''
    terms = in_channels

    max_val    = (1 << width_in) - 1       # Unsigned
    max_weight = (1 << (weight_width - 1))

    max_sum = terms * max_val * max_weight
    abs_bits = max_sum.bit_length()
    return str(abs_bits + 1)   # +1 for sign bit

def unpack_weights(packed_val: int, WW: int, OC: int, IC: int):
    """Reconstructs the 4D weights matrix from the Verilog parameter integer."""
    mask = (1 << WW) - 1
    sign_bit = 1 << (WW - 1)
    
    weights2 = []
    bit_shift = 0
    
    # Must mirror the exact same LSB -> MSB iteration order used in packing
    for _ in range(OC):
        oc_list = []
        for _ in range(IC):
            # Extract the specific bits for this weight
            w_bits = (packed_val >> bit_shift) & mask
            
            # Convert from two's complement back to a signed Python integer
            if w_bits & sign_bit:
                w = w_bits - (1 << WW)
            else:
                w = w_bits
                
            oc_list.append(w)
            bit_shift += WW
        weights2.append(oc_list)
        
    return weights2

def unpack_biases(packed_val: int, BW: int, OC: int):
    """Reconstructs the 4D weights matrix from the Verilog parameter integer."""
    mask = (1 << BW) - 1
    sign_bit = 1 << (BW - 1)
    
    biases1 = []
    bit_shift = 0
    
    # Must mirror the exact same LSB -> MSB iteration order used in packing
    for _ in range(OC):
        # Extract the specific bits for this weight
        w_bits = (packed_val >> bit_shift) & mask
        
        # Convert from two's complement back to a signed Python integer
        if w_bits & sign_bit:
            w = w_bits - (1 << BW)
        else:
            w = w_bits
            
        biases1.append(w)
        bit_shift += BW
        
    return biases1

def linear_reference(weights_2d, biases_1d, InChannels, OutChannels):
    # One output channel instantiates a single neuron
    linear = nn.Linear(in_features=InChannels, out_features=OutChannels, bias=True)

    # Disable gradient tracking
    linear.weight.requires_grad = False
    linear.bias.requires_grad = False

    # Convert to float32 for deterministic conv math
    linear.weight.data = torch.tensor(weights_2d, dtype=torch.float32)
    linear.bias.data   = torch.tensor(biases_1d, dtype=torch.float32)

    return linear

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

class LinearLayerModel():
    def __init__(self, dut, weights: List[List[int]], biases: List[int], torch_ref=None):
        self._dut = dut
        self._InBits  = int(dut.InBits.value)
        self._OutBits = int(dut.OutBits.value)
        self._InChannels  = int(dut.InChannels.value)
        self._OutChannels = int(dut.OutChannels.value)

        self.w = np.array(weights, dtype=int)
        self.b = np.array(biases, dtype=int)
        self._torch_ref = torch_ref # Optional nn.Linear reference

    def consume(self):
        """Called by ModelRunner on input handshake."""
        # Unpack bits from pins using generic utility
        packed = int(self._dut.data_i.value)
        raw_inputs = unpack_terms(packed, self._InBits, self._InChannels)

        # 1. Handle BNN mapping if InBits == 1
        if self._InBits == 1:
            input_vals = [1 if x == 1 else -1 for x in raw_inputs]
        else:
            input_vals = raw_inputs

        # 2. Compute expected output (Standard Math)
        expected = []
        for oc in range(self._OutChannels):
            acc = int(self.b[oc])
            for ic in range(self._InChannels):
                acc += int(self.w[oc][ic]) * input_vals[ic]
            expected.append(acc)

        # 3. Optional: Cross-check against Torch to prevent "Model Drift"
        if self._torch_ref is not None:
            data_i_ref = torch.tensor(input_vals, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                torch_out = self._torch_ref(data_i_ref).squeeze(0).numpy().astype(int)
            
            # Wrap to match hardware bit-width for comparison
            mask = (1 << self._OutBits) - 1
            for i in range(self._OutChannels):
                t_wrapped = sign_extend(int(torch_out[i]) & mask, self._OutBits)
                e_wrapped = sign_extend(expected[i] & mask, self._OutBits)
                assert t_wrapped == e_wrapped, f"Torch Drift! ch{i}: {e_wrapped} vs {t_wrapped}"

        return tuple(expected)

    def produce(self, expected):
        """Called by ModelRunner on output handshake."""
        got_raw = unpack_terms(int(self._dut.data_o.value), self._OutBits, self._OutChannels)
        
        for ch in range(self._OutChannels):
            got = got_raw[ch]
            exp = sign_extend(expected[ch] & ((1 << self._OutBits) - 1), self._OutBits)
            assert got == exp, f"Mismatch ch{ch}: expected {exp}, got {got}"
    
class RandomDataGenerator:
    def __init__(self, dut):
        self._width_p = int(dut.InBits.value)
        self._InChannels = int(dut.InChannels.value)

    def generate(self):
        din = gen_input_channels(self._width_p, self._InChannels)
        packed = pack_terms(din, self._width_p)
        return packed, din

class OutputModel():
    def __init__(self, dut, g, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.valid_i, dut.ready_o)

        self._rv_out = ReadyValidInterface(self._clk_i, self._rst_i,
                                           dut.valid_o, dut.ready_i)
        self._generator = g
        self._length = l

        self._coro = None

        self._nout = 0

    def start(self):
        """ Start Output Model """
        if self._coro is not None:
            raise RuntimeError("Output Model already started")
        self._coro = cocotb.start_soon(self._run())

    def stop(self) -> None:
        """ Stop Output Model """
        if self._coro is None:
            raise RuntimeError("Output Model never started")
        self._coro.kill()
        self._coro = None

    async def wait(self, t):
        if self._coro is None:
            raise RuntimeError("Output Model never started")
        await with_timeout(self._coro, t, 'ns')

    def nproduced(self):
        return self._nout

    async def _run(self):
        """ Output Model Coroutine"""

        self._nout = 0
        clk_i = self._clk_i
        ready_i = self._dut.ready_i
        rst_i = self._dut.rst_i
        valid_o = self._dut.valid_o

        await FallingEdge(clk_i)

        if(not (rst_i.value.is_resolvable and rst_i.value == 0)):
            await FallingEdge(rst_i)

        # Precondition: Falling Edge of Clock
        while self._nout < self._length:
            consume = self._generator.generate()
            success = 0
            ready_i.value = consume

            # Wait until valid
            while(consume and not success):
                await RisingEdge(clk_i)
                assert_resolvable(valid_o)
                #assert valid_o.value.is_resolvable, f"Unresolvable value in valid_o (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."

                success = True if ready_i.value == 1 and valid_o.value == 1 else False
                if (success):
                    self._nout += 1

            await FallingEdge(clk_i)
        return self._nout

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
    linear = linear_reference(weights_2d, biases_1d, IC, OC)
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
    linear = linear_reference(weights_2d, biases_1d, IC, OC)
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