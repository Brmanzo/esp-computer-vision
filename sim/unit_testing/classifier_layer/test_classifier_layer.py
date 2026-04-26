# test_classifier_layer.py
import numpy as np
import os
from   pathlib import Path
import pytest
import shutil
import torch
from   torch  import nn
from   typing import List


from util.utilities  import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from util.bitwise    import unpack_terms, pack_terms, unpack_weights, unpack_biases
from util.gen_inputs import gen_weights, gen_biases, gen_input_channels

tbpath = Path(__file__).parent

import cocotb
from   cocotb.utils import get_sim_time
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

def torch_classifier_reference(sequence, weights, biases, in_ch, out_ch):
    """
    Performs full pipeline: Global Max Pool -> Linear -> Argmax.
    sequence: Shape (TermCount, InChannels)
    """
    with torch.no_grad():
        # Convert buffer to tensor [TermCount, InChannels]
        t_in = torch.tensor(sequence, dtype=torch.float32)
        
        # 1. Global Max Pool (across the time/sequence dimension)
        # Returns shape [1, InChannels]
        t_max = torch.amax(t_in, dim=0, keepdim=True)

        # 2. Linear Layer
        ref = nn.Linear(in_ch, out_ch, bias=True)
        ref.weight.data = torch.tensor(weights, dtype=torch.float32)
        ref.bias.data = torch.tensor(biases, dtype=torch.float32)
        
        logits = ref(t_max)
        class_id = torch.argmax(logits, dim=1).item()
        
        return class_id, logits.squeeze().tolist()

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

class ClassifierLayerModel:
    def __init__(self, dut, weights: List[List[int]], biases: List[int]):
        self._dut = dut
        self._data_i = dut.data_i
        self._class_o = dut.class_o

        # Global Max parameters
        self._term_bits = int(dut.TermBits.value)
        self._term_count = int(dut.TermCount.value)
        self._term_counter = 0
        self._current_max = None

        # Linear Parameters
        self._in_channels  = int(dut.InChannels.value)
        self._class_count = int(dut.ClassCount.value)

        # 2D array storing all weights in each filter: [OC][IC]
        self.w = np.array(weights, dtype=int)
        assert self.w.shape == (self._class_count, self._in_channels)

        # 1D array storing bias for each output channel: [OC]
        self.b = np.array(biases, dtype=int)

        # If scalar bias for OC=1, normalize to shape (1,)
        if self.b.shape == () and self._class_count == 1:
            self.b = self.b.reshape((1,))

        assert self.b.shape == (self._class_count,), (
            f"Bias shape mismatch: got {self.b.shape}, expected ({self._class_count},). "
            f"biases={biases!r}"
        )

        self._sequence_buffer = []

    def consume(self):
        assert_resolvable(self._data_i)
        packed_in = int(self._data_i.value.integer)
        x = unpack_terms(packed_in, self._term_bits, self._in_channels)

        # 1. Map values for 1-bit logic (0 -> -1) before buffering
        current_vector = [(1 if v == 1 else -1) if self._term_bits == 1 else v for v in x]
        self._sequence_buffer.append(current_vector)
        self._term_counter += 1

        # 2. When the full image sequence is collected
        if self._term_counter == self._term_count:
            
            # --- BRAIN A: Torch Reference ---
            torch_id, torch_logits = torch_classifier_reference(
                self._sequence_buffer, self.w, self.b, 
                self._in_channels, self._class_count
            )

            # --- BRAIN B: Internal Python Model (Manual MAC) ---
            # Manual Max Pool
            manual_max = [max(col) for col in zip(*self._sequence_buffer)]
            
            # Manual Linear Projection
            manual_logits = []
            for oc in range(self._class_count):
                acc = self.b[oc]
                for ic in range(self._in_channels):
                    acc += self.w[oc][ic] * manual_max[ic]
                manual_logits.append(acc)
            
            manual_id = manual_logits.index(max(manual_logits))

            # --- CROSS-CHECK A vs B ---
            # Use math.isclose or np.allclose if using floats, 
            # but for integers, exact equality is expected.
            assert manual_id == torch_id, (
                f"Reference Mismatch! Internal Model: {manual_id}, Torch: {torch_id}. "
                f"Check for tie-breaking or precision drift."
            )

            # Clean up for next handshake
            self._term_counter = 0
            self._sequence_buffer = []

            # Return the golden data for produce() to check against BRAIN C (the DUT)
            return [(torch_id, torch_logits)]

        return None

    def produce(self, expected):
        assert_resolvable(self._class_o)

        expected_id, expected_logits = expected
        got_id = int(self._class_o.value.integer)

        print(
            f"Produced class {got_id}, expected {expected_id}"
            f"logits={expected_logits} at time {get_sim_time(units='ns')}ns"
        )

        assert got_id == expected_id, (
            f"Class mismatch. Expected {expected_id}, got {got_id}"
        )

class RandomDataGenerator:
    def __init__(self, dut):
        self._width_p = int(dut.TermBits.value)
        self._InChannels = int(dut.InChannels.value)

    def generate(self):
        raw_din = gen_input_channels(self._width_p, self._InChannels)
        packed_din = pack_terms(raw_din, self._width_p)
        return (packed_din, raw_din)

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
