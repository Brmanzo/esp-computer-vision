# test_classifier_layer.py
import shutil

import git
import os
import sys
import numpy as np
import queue
from itertools import product
from decimal import Decimal

# I don't like this, but it's convenient.
_REPO_ROOT = git.Repo(search_parent_directories=True).working_tree_dir
assert _REPO_ROOT is not None, "REPO_ROOT path must not be None"
assert (os.path.exists(_REPO_ROOT)), "REPO_ROOT path must exist"
sys.path.append(os.path.join(_REPO_ROOT, "util"))
from utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence, delay_cycles, ReadyValidInterface
tbpath = os.path.dirname(os.path.realpath(__file__))

import pytest

import cocotb
from typing import List

from cocotb.utils import get_sim_time
from cocotb.triggers import Timer, RisingEdge, FallingEdge, with_timeout
from cocotb.result import SimTimeoutError
   
import random
random.seed(42)

timescale = "1ps/1ps"

def sign_extend(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    sign_bit = 1 << (width - 1)
    return (value ^ sign_bit) - sign_bit

def pack(inputs, term_bits):
    packed = 0
    mask = (1 << term_bits) - 1
    for i, x in enumerate(inputs):
        packed |= (x & mask) << (i * term_bits)
    return packed

def unpack(packed, term_bits, input_count):
    unpacked = []
    mask = (1 << term_bits) - 1
    for i in range(input_count):
        raw = (packed >> (i * term_bits)) & mask
        unpacked.append(sign_extend(raw, term_bits))
    return unpacked

def trunc_signed(value: int, width: int) -> int:
    return sign_extend(value, width)

def gen_weights(IC: int, OC: int, WW: int, seed: int | None = None):
    rng = random.Random(seed)
    if WW < 2:
        raise ValueError("Weight width must be at least 2 to include negative values in test kernels.")
    elif WW == 2:
        rand_weight_value = lambda: rng.choice([-1, 0, 1])
    else:
        max_val = (1 << (WW - 1)) - 1
        min_val = -(1 << (WW - 1))
        rand_weight_value = lambda: rng.randint(min_val, max_val)

    # Generate the 2D matrix
    weights2 = [[rand_weight_value() for _ in range(IC)] for _ in range(OC)]

    # Pack the 2D matrix into a single integer for the Verilog Parameter
    packed_weights = 0
    bit_shift = 0
    mask = (1 << WW) - 1

    # Iterate from LSB to MSB: cols -> rows -> InChannels -> OutChannels
    for oc in range(OC):
        for ic in range(IC):
            w = weights2[oc][ic]

            w_bits = w if w >= 0 else (1 << WW) + w
            w_bits = w_bits & mask # Ensure it fits within WW bits
            
            # Shift and combine into the main bit vector
            packed_weights |= (w_bits << bit_shift)
            bit_shift += WW

    return packed_weights

def gen_biases(BW: int, OC: int, seed: int | None = None):
    rng = random.Random(seed)

    max_val = (1 << (BW - 1)) - 1
    min_val = -(1 << (BW - 1))
    rand_bias_value = lambda: rng.randint(min_val, max_val)

    # Generate the 2D matrix
    biases1 = [rand_bias_value()  for _ in range(OC)]

    # Pack the 2D matrix into a single integer for the Verilog Parameter
    packed_biases = 0
    bit_shift = 0
    mask = (1 << BW) - 1

    # Iterate from LSB to MSB: cols -> rows -> InChannels -> OutChannels
    for oc in range(OC):
        w = biases1[oc]

        w_bits = w if w >= 0 else (1 << BW) + w
        w_bits = w_bits & mask # Ensure it fits within BW bits
        
        # Shift and combine into the main bit vector
        packed_biases |= (w_bits << bit_shift)
        bit_shift += BW

    return packed_biases

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

def pack_data_i(samples, width):
    """
    samples: list[int] length = InChannels
    width: InBits
    Returns packed int: sum(samples[ic] << (ic*width))
    """
    mask = (1 << width) - 1
    out = 0
    for ic, v in enumerate(samples):
        out |= (int(v) & mask) << (ic * width)
    return out

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

        self._expected = queue.SimpleQueue()

    def consume(self):
        assert_resolvable(self._data_i)

        packed_in = int(self._data_i.value.integer)
        
        # 1. Unpack incoming feature map (InChannels wide)
        x = unpack(packed_in, self._term_bits, self._in_channels)

        # 2. Global Max Pooling (across spatial TermCount)
        if self._term_counter == 0:
            self._current_max = x[:]
        else:
            for ch in range(self._in_channels):
                self._current_max[ch] = max(self._current_max[ch], x[ch])

        self._term_counter += 1

        # 3. When spatial max is complete, run the linear layer and comparator!
        if self._term_counter == self._term_count:
            
            # --- Linear Layer MAC Operation ---
            expected_logits = [0 for _ in range(self._class_count)]
            for oc in range(self._class_count):
                acc = self.b[oc]
                for ic in range(self._in_channels):
                    acc += self.w[oc][ic] * self._current_max[ic]
                expected_logits[oc] = acc

            # --- Comparator (Argmax) Operation ---
            max_val = max(expected_logits)
            class_id = expected_logits.index(max_val)  # lowest index wins ties

            # Queue the final class ID and the logits for debug printing
            self._expected.put((class_id, expected_logits[:]))
            
            # Reset for the next image
            self._term_counter = 0
            self._current_max = None

    def produce(self):
        assert_resolvable(self._class_o)

        expected_id, expected_logits = self._expected.get()
        got_id = int(self._class_o.value.integer)

        print(
            f"Produced class {got_id}, expected {expected_id}, "
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
        # Generates a clean list of ints, one for each channel
        return [random.randint(0, (1 << self._width_p) - 1)
                for _ in range(self._InChannels)]

class RateGenerator():
    def __init__(self, dut, r):
        self._rate = r

    def generate(self):
        if(self._rate == 0):
            return False
        else:
            return (random.randint(1,int(1/self._rate)) == 1)

class OutputModel():
    def __init__(self, dut, g, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut

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

                success = True if (valid_o.value == 1) else False
                if (success):
                    self._nout += 1

            await FallingEdge(clk_i)
        return self._nout

class InputModel():
    def __init__(self, dut, data, rate, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut

        self._rate = rate
        self._data = data
        self._length = l

        self._coro = None

        self._nin = 0

    def start(self):
        """ Start Input Model """
        if self._coro is not None:
            raise RuntimeError("Input Model already started")
        self._coro = cocotb.start_soon(self._run())

    def stop(self) -> None:
        """ Stop Input Model """
        if self._coro is None:
            raise RuntimeError("Input Model never started")
        self._coro.kill()
        self._coro = None

    async def wait(self, t):
        if self._coro is None:
            raise RuntimeError("Input Model never started")
        await with_timeout(self._coro, t, 'ns')

    def nconsumed(self):
        return self._nin

    async def _run(self):
        """ Input Model Coroutine"""

        self._nin = 0
        clk_i = self._clk_i
        rst_i = self._dut.rst_i
        ready_o = self._dut.ready_o
        valid_i = self._dut.valid_i
        data_i = self._dut.data_i
        
        # 1. FIX: Grab the bit width from the DUT (assuming TermBits for classifier)
        w = int(self._dut.TermBits.value)

        await delay_cycles(self._dut, 1, False)

        if(not (rst_i.value.is_resolvable and rst_i.value == 0)):
            await FallingEdge(rst_i)

        await delay_cycles(self._dut, 2, False)

        # 2. FIX: Generate the very first data sample before entering the loop
        din = self._data.generate()

        # Precondition: Falling Edge of Clock
        while self._nin < self._length:
            produce = bool(self._rate.generate())
            valid_i.value = int(produce)

            # 1. Pack the list of integers into a single big integer
            packed_din = pack_data_i(din, w)

            # 2. Assign the packed integer to the pin!
            data_i.value = packed_din if produce else 0

            # Wait for handshake if producing, otherwise just advance a cycle
            success = False

            # Wait until ready
            while(produce and not success):
                await RisingEdge(clk_i)
                assert_resolvable(ready_o)

                success = True if (ready_o.value == 1) else False
                if (success):
                    self._nin += 1
                    
                    # 3. FIX: Generate the NEXT data sample only after a successful transfer!
                    din = self._data.generate()

            await FallingEdge(clk_i)
            
        # Optional but recommended: Drop valid to 0 when finished
        valid_i.value = 0
        return self._nin

class ModelRunner():
    def __init__(self, dut, model):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i

        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.valid_i, dut.ready_o)
        self._rv_out = ReadyValidInterface(self._clk_i, self._rst_i,
                                           dut.valid_o, dut.ready_i)

        self._model = model
        self._coro_run_input = None
        self._coro_run_output = None

    def start(self):
        if self._coro_run_input is not None:
            raise RuntimeError("Model already started")
        self._coro_run_input = cocotb.start_soon(self._run_input(self._model))
        self._coro_run_output = cocotb.start_soon(self._run_output(self._model))

    async def _run_input(self, model):
        while True:
            await self._rv_in.handshake(None)
            self._model.consume()

    async def _run_output(self, model):
        while True:
            await self._rv_out.handshake(None)
            self._model.produce()

    def stop(self):
        if self._coro_run_input is None or self._coro_run_output is None:
            raise RuntimeError("Monitor never started")
        self._coro_run_input.kill()
        self._coro_run_output.kill()
        self._coro_run_input = None
        self._coro_run_output = None

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
    T = int(dut.TermCount.value)
    groups = 10
    l_in = groups * T
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
            f"with TermCount={T}, in_rate={in_rate}, out_rate={out_rate}"
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
