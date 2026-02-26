import git
import os
import sys
import git
import queue
import math
import numpy as np
from typing import List, Optional
from functools import reduce
import torch
import torch.nn.functional as F

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

def output_width(width_in: int, weight_width: int, kernel_width: int=3, in_channels: int=1) -> str:
    '''Calculates proper output width for given input width amount of accumulations.'''
    kernel_area = kernel_width * kernel_width
    terms = kernel_area * in_channels

    max_val    = (1 << width_in) - 1       # Unsigned``
    max_weight = (1 << (weight_width - 1)) # Signed

    max_sum = terms * max_val * max_weight
    abs_bits = max_sum.bit_length()
    return str(abs_bits + 1)   # +1 for sign bit

def pack_weights_in(weights_4d, width, kernel_area, IC):
    """
    weights_4d: [OC][IC][kernel_area] values (signed ints)
    Packing order: k fastest, then ic, then oc
      shift = width * (k + kernel_area*(ic + IC*oc))
    """
    mask = (1 << width) - 1
    lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1

    out = 0
    for oc, ws_by_ic in enumerate(weights_4d):
        assert len(ws_by_ic) == IC
        for ic, ws in enumerate(ws_by_ic):
            assert len(ws) == kernel_area
            for k, w in enumerate(ws):
                assert lo <= w <= hi
                shift = width * (k + kernel_area * (ic + IC * oc))
                out |= (w & mask) << shift
    return out

def sign_extend(val: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (val ^ sign) - sign

def gen_kernels(WW: int, OC: int, IC: int, K: int, seed: int | None = None):
    rng = random.Random(seed)
    if WW < 2:
        raise ValueError("Weight width must be at least 2 to include negative values in test kernels.")
    elif WW == 2:
        rand_kernel_value = lambda: rng.choice([-1, 0, 1])
    else:
        max_val = (1 << (WW - 1)) - 1
        min_val = -(1 << (WW - 1))
        rand_kernel_value = lambda: rng.randint(min_val, max_val)

    # 1. Generate the 4D matrix
    kernels4 = [
        [
            [[rand_kernel_value() for _ in range(K)] for _ in range(K)]
            for _ in range(IC)
        ]
        for _ in range(OC)
    ]

    # 2. Pack the 4D matrix into a single integer for the Verilog Parameter
    packed_weights = 0
    bit_shift = 0
    mask = (1 << WW) - 1

    # Iterate from LSB to MSB: cols -> rows -> InChannels -> OutChannels
    for oc in range(OC):
        for ic in range(IC):
            for r in range(K):
                for c in range(K):
                    w = kernels4[oc][ic][r][c]
                    
                    # Convert negative values to two's complement representation
                    w_bits = w if w >= 0 else (1 << WW) + w
                    w_bits = w_bits & mask # Ensure it fits within WW bits
                    
                    # Shift and combine into the main bit vector
                    packed_weights |= (w_bits << bit_shift)
                    bit_shift += WW

    # Return as an integer (cocotb-runner handles integer parameters well)
    # We no longer need kernels_flat unless you use it elsewhere
    return packed_weights

def unpack_weights(packed_val: int, WW: int, OC: int, IC: int, K: int):
    """Reconstructs the 4D weights matrix from the Verilog parameter integer."""
    mask = (1 << WW) - 1
    sign_bit = 1 << (WW - 1)
    
    kernels4 = []
    bit_shift = 0
    
    # Must mirror the exact same LSB -> MSB iteration order used in packing
    for oc in range(OC):
        oc_list = []
        for ic in range(IC):
            ic_list = []
            for r in range(K):
                r_list = []
                for c in range(K):
                    # Extract the specific bits for this weight
                    w_bits = (packed_val >> bit_shift) & mask
                    
                    # Convert from two's complement back to a signed Python integer
                    if w_bits & sign_bit:
                        w = w_bits - (1 << WW)
                    else:
                        w = w_bits
                        
                    r_list.append(w)
                    bit_shift += WW
                ic_list.append(r_list)
            oc_list.append(ic_list)
        kernels4.append(oc_list)
        
    return kernels4

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

def to_torch_input(input_activation):
    # input_activation: [IC][H][W]
    x = torch.tensor(input_activation, dtype=torch.int32)   # (IC,H,W)
    x = x.unsqueeze(0).to(torch.int32)                      # (1,IC,H,W)
    return x

def to_torch_weights(kernels4):
    # kernels4: [OC][IC][K][K]
    w = torch.tensor(kernels4, dtype=torch.int32)           # (OC,IC,K,K)
    return w

def torch_conv_ref(input_activation, kernels4, stride):
    x = to_torch_input(input_activation).to(torch.float32)
    w = to_torch_weights(kernels4).to(torch.float32)
    y = F.conv2d(x, w, stride=stride, padding=0)            # (1,OC,H_out,W_out)
    return y.squeeze(0)

def unpack_data_i(packed, width_in, IC):
    mask = (1 << width_in) - 1
    return [ (packed >> (ic * width_in)) & mask for ic in range(IC) ]

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("WidthIn, WeightWidth, WidthOut, KernelWidth, InChannels, OutChannels, Weights", 
                         [(1, 2, output_width(1, 2, 3), 3, 1, 1, gen_kernels(2, 1, 1, 3, seed=1234)), # Intended Size
                          (2, 3, output_width(2, 3, 3), 3, 1, 1, gen_kernels(3, 1, 1, 3, seed=1234)), # Unsigned data_i
                          (4, 5, output_width(4, 5, 3), 3, 1, 1, gen_kernels(5, 1, 1, 3, seed=1234)),
                          (8, 8, output_width(8, 8, 3), 3, 1, 1, gen_kernels(8, 1, 1, 3, seed=1234))])
def test_width(test_name, simulator, WidthIn, WeightWidth, WidthOut, KernelWidth, InChannels, OutChannels, Weights):
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    
    # 1. Remove Weights from the CLI parameters dict so cocotb-runner doesn't pass it
    del parameters['Weights'] 
    
    param_str = f"WidthIn_{WidthIn}_WeightWidth_{WeightWidth}_WidthOut_{WidthOut}_test_{test_name}"
    custom_work_dir = os.path.join(tbpath, "run", "width", param_str, simulator)
    os.makedirs(custom_work_dir, exist_ok=True)
    
    # 2. Calculate total bits to format the Verilog hex string correctly
    total_bits = OutChannels * InChannels * (KernelWidth**2) * WeightWidth
    
    # 3. Write the massive integer as a strictly sized hex literal to a header file
    vh_path = os.path.join(custom_work_dir, "injected_weights.vh")
    with open(vh_path, "w") as f:
        hex_width = (total_bits + 3) // 4
        f.write(f"localparam logic signed [{total_bits-1}:0] INJECTED_WEIGHTS = {total_bits}'h{Weights:0{hex_width}x};\n")

    # Pass the massive integer strictly through the OS environment to Cocotb
    os.environ["INJECTED_WEIGHTS_INT"] = str(Weights)

    # Define the wrapper path and pass the extra arguments to your runner
    wrapper_path = os.path.join(tbpath, "tb_conv_layer.sv")

    runner(
        simulator=simulator, 
        timescale=timescale, 
        tbpath=tbpath, 
        params=parameters, 
        testname=test_name, 
        work_dir=custom_work_dir,
        includes=[custom_work_dir],          # Tells simulator where to find injected_weights.vh
        toplevel_override="tb_conv_layer",   # Forces simulator to use the wrapper as top-level
        extra_sources=[wrapper_path]         # Adds the wrapper file to the compilation list
    )
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
# Added 'Weights' to the tuple list, generating it with WW=2, OC=1, IC=1
@pytest.mark.parametrize("WidthOut, KernelWidth, Stride, LineWidthPx, LineCountPx, Weights", 
                         [(output_width(1, 2, 2), 2, 2, 16, 12, gen_kernels(2, 1, 1, 2, seed=1234)),
                          (output_width(1, 2, 4), 4, 4, 16, 12, gen_kernels(2, 1, 1, 4, seed=1234)),
                          (output_width(1, 2, 5), 5, 2, 17, 13, gen_kernels(2, 1, 1, 5, seed=1234))
                          ])

def test_stride(test_name, simulator, WidthOut, KernelWidth, Stride, LineWidthPx, LineCountPx, Weights):
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    
    # Remove Weights from CLI
    del parameters['Weights'] 
    
    param_str = f"KernelWidth_{KernelWidth}_Stride_{Stride}_LineWidthPx_{LineWidthPx}_LineCountPx_{LineCountPx}_test_{test_name}"
    custom_work_dir = os.path.join(tbpath, "run", "stride", param_str, simulator)
    os.makedirs(custom_work_dir, exist_ok=True)

    # Hardcode the DUT defaults since they aren't parametrized here
    WW, IC, OC = 2, 1, 1
    total_bits = OC * IC * (KernelWidth**2) * WW
    
    # Write the header file
    vh_path = os.path.join(custom_work_dir, "injected_weights.vh")
    with open(vh_path, "w") as f:
        hex_width = (total_bits + 3) // 4
        f.write(f"localparam logic signed [{total_bits-1}:0] INJECTED_WEIGHTS = {total_bits}'h{Weights:0{hex_width}x};\n")

    # Route weights via OS environment
    os.environ["INJECTED_WEIGHTS_INT"] = str(Weights)
    wrapper_path = os.path.join(tbpath, "tb_conv_layer.sv")

    # Run with wrapper hooks
    runner(
        simulator=simulator, 
        timescale=timescale, 
        tbpath=tbpath, 
        params=parameters, 
        testname=test_name, 
        work_dir=custom_work_dir,
        includes=[custom_work_dir],
        toplevel_override="tb_conv_layer",
        extra_sources=[wrapper_path]
    )

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
# Expanded to include WidthIn, WeightWidth, KernelWidth so the DUT scales with the Python math
@pytest.mark.parametrize("WidthIn, WeightWidth, KernelWidth, WidthOut, InChannels, OutChannels, Weights", 
                         [(1, 2, 3, output_width(1, 2, 3), 1, 2, gen_kernels(2, 2, 1, 3, seed=1234)),
                          (2, 3, 3, output_width(2, 3, 3), 2, 3, gen_kernels(3, 3, 2, 3, seed=1234)),
                          (4, 5, 3, output_width(4, 5, 3), 4, 5, gen_kernels(5, 5, 4, 3, seed=1234)),
                          (8, 8, 3, output_width(8, 8, 3), 8, 8, gen_kernels(8, 8, 8, 3, seed=1234))
                          ])

def test_channels(test_name, simulator, WidthIn, WeightWidth, KernelWidth, WidthOut, InChannels, OutChannels, Weights):
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    del parameters['Weights'] 
    
    param_str = f"InChannels_{InChannels}_OutChannels_{OutChannels}_test_{test_name}"
    custom_work_dir = os.path.join(tbpath, "run", "channels", param_str, simulator)
    os.makedirs(custom_work_dir, exist_ok=True)

    # Calculate total bits using the explicit parameters
    total_bits = OutChannels * InChannels * (KernelWidth**2) * WeightWidth
    
    # Write the header file
    vh_path = os.path.join(custom_work_dir, "injected_weights.vh")
    with open(vh_path, "w") as f:
        hex_width = (total_bits + 3) // 4
        f.write(f"localparam logic signed [{total_bits-1}:0] INJECTED_WEIGHTS = {total_bits}'h{Weights:0{hex_width}x};\n")

    # Route weights via OS environment
    os.environ["INJECTED_WEIGHTS_INT"] = str(Weights)
    wrapper_path = os.path.join(tbpath, "tb_conv_layer.sv")

    # Run with wrapper hooks
    runner(
        simulator=simulator, 
        timescale=timescale, 
        tbpath=tbpath, 
        params=parameters, 
        testname=test_name, 
        work_dir=custom_work_dir,
        includes=[custom_work_dir],
        toplevel_override="tb_conv_layer",
        extra_sources=[wrapper_path]
    )

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("LineWidthPx, WidthIn, WidthOut", [("16", "1", output_width(1, 2, 3))])
def test_lint(simulator, LineWidthPx, WidthIn, WidthOut):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("LineWidthPx, WidthIn, WidthOut", [("16", "1", output_width(1, 2))])
def test_style(simulator, LineWidthPx, WidthIn, WidthOut):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

class ConvLayerModel():
    def __init__(self, dut, weights: List[List[List[List[int]]]], output_activation: Optional[List[List[List[int]]]] = None):
        self._kernel_width = int(dut.KernelWidth.value)
        self._f = np.ones((self._kernel_width,self._kernel_width), dtype=int)
        self._dut = dut
        self._data_o = dut.data_o
        self._data_i = dut.data_i

        self._q = queue.SimpleQueue()

        self._input_width = int(dut.LineWidthPx.value)
        self._input_height = int(dut.LineCountPx.value)
        self._WidthOut = int(dut.WidthOut.value)
        self._InChannels  = int(dut.InChannels.value)
        self._OutChannels = int(dut.OutChannels.value)
        self._Stride = int(dut.Stride.value)

        self._output_activation = output_activation 
        self._r = 0
        self._c = 0
        self._OW = (self._input_width - self._kernel_width) // self._Stride + 1
        self._OH = (self._input_height - self._kernel_width) // self._Stride + 1

        # We're going to initialize _buf with NaN so that we can
        # detect when the output should be not an X in simulation
        # Buffer for all input channels, storing the most recent kernel_width values for each channel
        self._buf = [np.zeros((self._kernel_width,self._input_width))/0 for _ in range(self._InChannels)]
        self._deqs = 0
        self._enqs = 0

        # kernel 4D array storing all kernels in each filter: [OC][IC][K][K]
        self.k = np.array(weights, dtype=int)
        assert self.k.shape == (self._OutChannels, self._InChannels, self._kernel_width, self._kernel_width)
        self._in_idx = 0

        # Valid kernel positions within input image depending on stride and kernel size
        invalid_region = self._kernel_width - 1
        S = int(self._Stride)
        span_w = (self._input_width  - 1) - (self._kernel_width - 1)
        span_h = (self._input_height - 1) - (self._kernel_width - 1)

        assert span_w >= 0 and span_h >= 0, "Kernel exceeds image bounds"
        assert (span_w % S) == 0 and (span_h % S) == 0, "Invalid configuration: Stride does not tile evenly"
        
        self._valid_cycles = np.ones((self._input_height, self._input_width), dtype=bool)
        self._valid_cycles[:invalid_region, :] = False
        self._valid_cycles[:, :invalid_region] = False
        
        # If stride larger than normal, invalidate the skipped over elements
        if S > 1:
            for r in range(invalid_region, self._input_height):
                for c in range(invalid_region, self._input_width):
                    if ((r - invalid_region) % S) != 0 or ((c - invalid_region) % S) != 0:
                        self._valid_cycles[r, c] = False

    # Now let's scale this up a little bit
    # You can define functions to do the steps in convolution
    def update_window(self, buf, inp):
        temp = buf.flatten()

        # Now shift everything by 1
        temp = np.roll(temp, -1, axis=0)

        # Add the new input, replacing the input that was "kicked out"
        temp[-1] = inp

        # Now reshape it back into the original buffer
        temp = np.reshape(temp, buf.shape)
        buf = temp
        return buf

    def apply_kernel(self, bufs):
        result = np.zeros(self._OutChannels, dtype=int)

        windows = [b[:, -self._kernel_width:].astype(int, copy=False) for b in bufs]

        result = np.zeros(self._OutChannels, dtype=int)
        for oc in range(self._OutChannels):
            acc = 0
            for ic in range(self._InChannels):
                acc += int((self.k[oc, ic] * windows[ic]).sum())
            result[oc] = acc
        return result

    def consume(self):
        """Called on each INPUT handshake.
        Returns:
          - None if this input position should NOT produce an output
          - int expected value if it SHOULD produce an output
        """
        assert_resolvable(self._data_i)
        packed_data_i = []
        packed = int(self._data_i.value.integer)
        packed_data_i = unpack_data_i(packed, int(self._dut.WidthIn.value), self._InChannels)

        # advance windows on EVERY accepted input
        for ic, inp in enumerate(packed_data_i):
            self._buf[ic] = self.update_window(self._buf[ic], inp)

        idx = self._enqs
        x = idx % int(self._input_width)
        y = idx // int(self._input_width)
        self._enqs += 1

        if y >= int(self._input_height):
            return None

        if not self._valid_cycles[y, x]:
            return None

        # compute expected NOW, while _buf matches this accepted input position
        expected = self.apply_kernel(self._buf)
        return expected

    def produce(self, expected):
        assert_resolvable(self._data_o)

        w = int(self._dut.WidthOut.value)
        packed = int(self._data_o.value.integer)

        # Write all channels at the SAME (r,c)
        for ch in range(self._OutChannels):
            raw = (packed >> (ch * w)) & ((1 << w) - 1)   # ch0 in LSB slice assumption
            got = sign_extend(raw, w)
            exp = int(expected[ch])

            print(f"Output #{self._deqs} (r={self._r}, c={self._c}) ch{ch}: expected {exp}, got {got} (raw=0x{raw:x})")

            if self._output_activation is not None:
                self._output_activation[ch][self._r][self._c] = got

            assert got == exp, (
                f"Mismatch at output #{self._deqs} (r={self._r}, c={self._c}) ch{ch}: expected {exp}, got {got} "
                f"(raw=0x{raw:x})"
            )

        # Advance pixel position ONCE per output handshake/vector
        self._c += 1
        if self._c >= self._OW:
            self._c = 0
            self._r += 1
            if self._r >= self._OH:
                self._r = 0

class ReadyValidInterface():
    def __init__(self, clk, reset, ready, valid):
        self._clk_i = clk
        self._rst_i = reset
        self._ready = ready
        self._valid = valid

    def is_in_reset(self):
        if((not self._rst_i.value.is_resolvable) or self._rst_i.value  == 1):
            return True
        
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
        self._width_p = int(dut.WidthIn.value)
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
    def __init__(self, dut, data, rate, l, input_activation=None):
        self._clk_i = dut.clk_i
        self._rst_i = dut.rst_i
        self._dut = dut
        
        self._rv_in = ReadyValidInterface(self._clk_i, self._rst_i,
                                          dut.valid_i, dut.ready_o)

        self._rate = rate
        self._data = data
        self._length = l
        self._input_activation = input_activation

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

        # Geometry + channel count
        W  = int(self._dut.LineWidthPx.value)
        H  = int(self._dut.LineCountPx.value)
        IC = int(self._dut.InChannels.value)
        w  = int(self._dut.WidthIn.value)

        # Cursor for activation matrix (y,x)
        y = 0
        x = 0

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
                    # Record ONLY on accepted input
                    if self._input_activation is not None:
                        # Optional safety checks
                        assert len(din) == IC, f"din has {len(din)} chans, expected {IC}"
                        assert y < H and x < W, f"Activation cursor out of bounds (y={y}, x={x}, H={H}, W={W})"

                        for ic in range(IC):
                            self._input_activation[ic][y][x] = int(din[ic])

                    # Advance pixel cursor once per accepted input
                    x += 1
                    if x >= W:
                        x = 0
                        y += 1
                        if y >= H:
                            y = 0  # wrap (or raise if you expect exactly one frame)

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
            exp = self._model.consume()     # exp is None or int
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
    """Drive pixels until the first VALID kernel position, then expect 1 output."""

    W  = int(dut.LineWidthPx.value)
    K  = int(dut.KernelWidth.value)
    IC = int(dut.InChannels.value)
    OC = int(dut.OutChannels.value)
    WW = int(dut.WeightWidth.value)

    # Number of accepted inputs until first valid output position (x=K-1, y=K-1)
    N_first = (K - 1) * W + (K - 1) + 1

    # We expect exactly ONE output for this test (the first valid position)
    N_out = 1

    rate = 1

    packed_weights = int(os.environ["INJECTED_WEIGHTS_INT"])
    kernels_4d = unpack_weights(packed_weights, WW, OC, IC, K)
    
    model = ConvLayerModel(dut, kernels_4d)
    m = ModelRunner(dut, model)

    om = OutputModel(dut, RateGenerator(dut, 1), N_out)               # consume 1 output
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, rate), N_first)  # produce N_first inputs

    dut.ready_i.value = 0
    dut.valid_i.value = 0
    dut.data_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Wait until that single output is observed
    tmo_ns = 4 * N_first + 50

    timed_out = False
    try:
        await om.wait(tmo_ns)
    except SimTimeoutError:
        timed_out = True

    assert not timed_out, (
        f"Timed out waiting for first valid output. "
        f"W={W}, K={K}, expected after ~{N_first} accepted inputs."
    )

    dut.valid_i.value = 0
    dut.ready_i.value = 0

async def rate_tests(dut, in_rate, out_rate):
    W  = int(dut.LineWidthPx.value)
    H  = int(dut.LineCountPx.value)
    K  = int(dut.KernelWidth.value)
    IC = int(dut.InChannels.value)
    OC = int(dut.OutChannels.value)
    WW = int(dut.WeightWidth.value)
    S  = int(dut.Stride.value)

    # Observe H rows of VALID outputs
    invalid = K - 1
    N_in = W * H
    H_out = ((H - K) // S) + 1
    W_out = ((W - K) // S) + 1
    l_out = W_out * H_out   

    input_activation  = [[[0 for _ in range(W)] for _ in range(H)] for _ in range(IC)]
    output_activation = [[[0 for _ in range(W_out)] for _ in range(H_out)] for _ in range(OC)]

    # Consumer ready probability
    slow = min(in_rate, out_rate)  # bottleneck probability
    slow = max(slow, 0.05)         # avoid insane timeouts at tiny rates in fuzz

    first_out_wait_ns = int((2 * (K - 1) * W + 2 * (K - 1) + 200) / slow)
    timeout_ns        = int((H_out * N_in + 500) / slow)

    packed_weights = int(os.environ["INJECTED_WEIGHTS_INT"])
    kernels_4d = unpack_weights(packed_weights, WW, OC, IC, K)

    model = ConvLayerModel(dut, kernels_4d, output_activation)
    m = ModelRunner(dut, model)

    # Consumer fuzzed; producer always drives valid
    om = OutputModel(dut, RateGenerator(dut, out_rate), l_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, in_rate), N_in, input_activation)

    dut.ready_i.value = 0
    dut.valid_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # First output wait: producer is full rate, but DUT may stall due to consumer backpressure.
    # Give a bound proportional to N_first and 1/rate.
    # Wait until valid_o ever asserts (not necessarily handshake)
    try:
        await with_timeout(RisingEdge(dut.valid_o), first_out_wait_ns, 'ns')
    except SimTimeoutError:
        assert 0, (
            f"Timed out waiting for valid_o high. "
            f"W={W}, K={K}, S={S}, H_out={H_out}, W_out={W_out}, N_in={N_in}, waited={first_out_wait_ns} ns."
        )

    # Now wait for exactly l_out output handshakes
    try:
        await om.wait(timeout_ns)
        # --- Print input ---
        for ic in range(IC):
            print(f"\nInput Activation for IC{ic}")
            for r in range(H):
                print(" ".join(f"{input_activation[ic][r][c]:2d}" for c in range(W)))

        # --- Print kernels ---
        for oc in range(OC):
            print(f"\nKernel for OC{oc}")
            for ic in range(IC):
                print(f"  IC{ic}")
                for r in range(K):
                    print(" ".join(f"{kernels_4d[oc][ic][r][c]:4d}" for c in range(K)))

        # --- Print DUT-captured output (make sure output_activation is H_out x W_out) ---
        for oc in range(OC):
            print(f"\nOutput Activation (DUT) for OC{oc}")
            for r in range(H_out):
                print(" ".join(f"{output_activation[oc][r][c]:4d}" for c in range(W_out)))

        # Verify against PyTorch reference convolution
        ref = torch_conv_ref(input_activation, kernels_4d, S)  # (OC,H_out,W_out)

        for oc in range(OC):
            print(f"\nExpected (PyTorch) for OC{oc}")
            for r in range(H_out):
                print(" ".join(f"{int(ref[oc, r, c]):4d}" for c in range(W_out)))
        assert np.allclose(output_activation, ref.int().numpy()), "Output activation does not match PyTorch reference"
    except SimTimeoutError:
        assert 0, (
            f"Timed out. Expected {l_out} output handshakes "
            f"(W_out={W_out}, H_out={H_out}). Got {om.nproduced()} in {timeout_ns} ns. "
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
