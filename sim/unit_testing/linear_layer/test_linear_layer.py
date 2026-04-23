import os
import shutil
import sys
import git
import queue
import numpy as np
from typing import List
import torch
from torch import nn


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

from cocotb.triggers import RisingEdge, FallingEdge, with_timeout
from cocotb.result import SimTimeoutError

from cocotb_test.simulator import run
   
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

def sign_extend(val: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (val ^ sign) - sign

def gen_weights(WW: int, OC: int, IC: int, seed: int | None = None):
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

def unpack_data_i(packed, width_in, IC):
    mask = (1 << width_in) - 1
    return [ (packed >> (ic * width_in)) & mask for ic in range(IC) ]

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
        (1, 2,  1, output_width(1, 2, 1),    1, gen_weights(2,  1,  1, seed=1234), 2, gen_biases(2, 1)),
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
        self._data_o = dut.data_o
        self._data_i = dut.data_i

        self._q = queue.SimpleQueue()

        self._InBits     = int(dut.InBits.value)
        self._OutBits    = int(dut.OutBits.value)
        self._InChannels  = int(dut.InChannels.value)
        self._OutChannels = int(dut.OutChannels.value)

        # Buffer for all input channels, storing the most recent data_i values for each channel
        self._buf = [0 for _ in range(self._InChannels)]
        self._deqs = 0
        self._enqs = 0

        # 2D array storing all weights in each filter: [OC][IC]
        self.w = np.array(weights, dtype=int)
        assert self.w.shape == (self._OutChannels, self._InChannels)

        # 1D array storing bias for each output channel: [OC]
        self.b = np.array(biases, dtype=int)

        # If  scalar bias for OC=1, normalize to shape (1,)
        if self.b.shape == () and self._OutChannels == 1:
            self.b = self.b.reshape((1,))

        assert self.b.shape == (self._OutChannels,), (
            f"Bias shape mismatch: got {self.b.shape}, expected ({self._OutChannels},). "
            f"biases={biases!r}"
        )

        self._torch_ref = torch_ref
        self._W = torch.tensor(weights, dtype=torch.int64)   # [OC, IC]
        self._b = torch.tensor(biases, dtype=torch.int64)    # [OC]

    def consume(self):
        """Called on each INPUT handshake.
        Returns:
          - None if this input position should NOT produce an output
          - int expected value if it SHOULD produce an output
        """
        assert_resolvable(self._data_i)
        packed_data_i = []
        packed = int(self._data_i.value.integer)
        packed_data_i = unpack_data_i(packed, int(self._dut.InBits.value), self._InChannels)

        # load data_i into buffer for all channels
        for ic, inp in enumerate(packed_data_i):
            self._buf[ic] = inp
        self._enqs += 1

        # compute expected this cycle, while _buf matches this accepted input position
        expected = [0 for _ in range(self._OutChannels)]
        for oc in range(self._OutChannels):
            acc = self.b[oc]
            for ic in range(self._InChannels):
                acc += self.w[oc][ic] * self._buf[ic]
            expected[oc] = acc

        data_i_ref = torch.tensor(packed_data_i, dtype=torch.int64)
        data_o_ref = (self._W @ data_i_ref) + self._b
        mask = (1 << self._OutBits) - 1

        exp_wrapped = [sign_extend(int(v) & mask, self._OutBits) for v in expected]
        ref_wrapped = [sign_extend(int(data_o_ref[oc].item()) & mask, self._OutBits) for oc in range(self._OutChannels)]
        # Check that Pytorch is equal to our reference python model
        assert exp_wrapped == ref_wrapped, f"Expected {exp_wrapped}, got {ref_wrapped}"
        return expected

    def produce(self, expected):
        assert_resolvable(self._data_o)

        w = int(self._dut.OutBits.value)
        packed = int(self._data_o.value.integer)

        for ch in range(self._OutChannels):
            raw = (packed >> (ch * w)) & ((1 << w) - 1)
            got = sign_extend(raw, w)
            exp = int(expected[ch])

            print(f"Output #{self._deqs} ch{ch}: expected {exp}, got {got} (raw=0x{raw:x})")

            assert got == exp, (
                f"Mismatch at output #{self._deqs} ch{ch}: expected {exp}, got {got} (raw=0x{raw:x})"
            )

        self._deqs += 1

class ReadyValidInterface():
    def __init__(self, clk, reset, ready, valid):
        self._clk_i = clk
        self._rst_i = reset
        self._ready = ready
        self._valid = valid

    def is_in_reset(self):
        return (not self._rst_i.value.is_resolvable) or (int(self._rst_i.value) == 1)
        
    def assert_resolvable(self):
        if(not self.is_in_reset()):
            assert_resolvable(self._valid)
            assert_resolvable(self._ready)

    def is_handshake(self):
        return ((self._valid.value == 1) and (self._ready.value == 1))

    async def _handshake(self):
        while True:
            await RisingEdge(self._clk_i)
            if (not self.is_in_reset()):
                self.assert_resolvable()
                if(self.is_handshake()):
                    break

    async def handshake(self, ns):
        """Wait for a handshake, raising an exception if it hasn't
        happened after ns nanoseconds of simulation time"""

        # If ns is none, wait indefinitely
        if(ns):
            await with_timeout(self._handshake(), ns, 'ns')
        else:
            await self._handshake()    

class CountingDataGenerator():
    def __init__(self, dut):
        self._dut = dut
        self._cur = 0

    def generate(self):
        value = self._cur
        self._cur += 1
        return value
    
class RandomDataGenerator:
    def __init__(self, dut):
        self._width_p = int(dut.InBits.value)
        self._InChannels = int(dut.InChannels.value)

    def generate(self):
        return [random.randint(0, (1 << self._width_p) - 1)
                for _ in range(self._InChannels)]

class CountingGenerator():
    def __init__(self, dut, r):
        self._rate = int(1/r)
        self._init = 0

    def generate(self):
        if(self._rate == 0):
            return False
        else:
            retval = (self._init == 1)
            self._init = (self._init + 1) % self._rate
            return retval

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

class InputModel():
    def __init__(self, dut, data, rate, l):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.valid_i, dut.ready_o)

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
        if self._coro is not None:
            await with_timeout(self._coro, t, 'ns')

    def nconsumed(self):
        return self._nin

    async def _run(self):
        """Input Model Coroutine (records input_activation on handshake)."""

        self._nin = 0
        clk_i   = self._clk_i
        rst_i   = self._dut.rst_i
        ready_o = self._dut.ready_o
        valid_i = self._dut.valid_i
        data_i  = self._dut.data_i

        w  = int(self._dut.InBits.value)

        valid_i.value = 0
        data_i.value  = 0

        await delay_cycles(self._dut, 1, False)

        if not (rst_i.value.is_resolvable and rst_i.value == 0):
            await FallingEdge(rst_i)

        await delay_cycles(self._dut, 2, False)

        # Precondition: Falling edge of clock
        din = self._data.generate()  # list length IC

        while self._nin < self._length:
            produce = bool(self._rate.generate())
            valid_i.value = int(produce)

            # Drive current sample (hold stable until handshake)
            data_i.value = pack_data_i(din, w)

            # Wait for handshake if producing, otherwise just advance a cycle
            success = False
            while produce and not success:
                await RisingEdge(clk_i)
                assert_resolvable(ready_o)
                success = bool(valid_i.value) and bool(ready_o.value)

                if success:
                    # Next sample
                    din = self._data.generate()
                    self._nin += 1

            await FallingEdge(clk_i)

        # Stop driving
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
        self._events = queue.SimpleQueue()

        self._coro_run_in = None
        self._coro_run_out = None

    def start(self):
        """Start model"""
        if self._coro_run_in is not None or self._coro_run_out is not None:
            raise RuntimeError("Model already started")
        self._coro_run_in  = cocotb.start_soon(self._run_input())
        self._coro_run_out = cocotb.start_soon(self._run_output())

    async def _run_input(self):
        while True:
            await self._rv_in.handshake(None)
            exp = self._model.consume()
            if exp is not None:
                self._events.put(exp)

    async def _run_output(self):
        while True:
            await self._rv_out.handshake(None)
            assert self._events.qsize() > 0, "Error! Module produced output without expected input"
            expected = self._events.get()
            self._model.produce(expected)

    def stop(self):
        """Stop model"""
        if self._coro_run_in is None and self._coro_run_out is None:
            raise RuntimeError("Model never started")
        if self._coro_run_in is not None:
            self._coro_run_in.kill()
            self._coro_run_in = None
        if self._coro_run_out is not None:
            self._coro_run_out.kill()
            self._coro_run_out = None
    

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