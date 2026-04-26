import numpy as np
import os
from   pathlib import Path
import pytest
import queue
import torch
import torch.nn.functional as F
from   typing import List, Optional

from util.utilities  import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence
from util.bitwise    import sign_extend, pack_terms, unpack_terms, unpack_kernel_weights
from util.gen_inputs import gen_kernels, gen_input_channels
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel

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

def output_width(width_in: int, weight_width: int, kernel_width: int=3, in_channels: int=1) -> str:
    terms = kernel_width * kernel_width * in_channels

    # For signed inputs, the max magnitude is 2^(width-1)
    # e.g., for 2 bits, range is -2 to 1. Max magnitude is 2.
    max_val    = (1 << (width_in - 1)) 
    max_weight = (1 << (weight_width - 1))

    max_sum = terms * max_val * max_weight
    abs_bits = max_sum.bit_length()
    return str(abs_bits + 1)

def to_torch_input(input_activation):
    # input_activation: [IC][H][W]
    x = torch.tensor(input_activation, dtype=torch.int32)   # (IC,H,W)
    x = x.unsqueeze(0).to(torch.int32)                      # (1,IC,H,W)
    return x

def to_torch_weights(kernels4):
    # kernels4: [OC][IC][K][K]
    w = torch.tensor(kernels4, dtype=torch.int32)           # (OC,IC,K,K)
    return w

def torch_conv_ref(input_activation, kernels4, stride, in_bits=1, out_bits=1):
    x = to_torch_input(input_activation).to(torch.float32)
    w = to_torch_weights(kernels4).to(torch.float32)

    if in_bits == 1:
        # Match MAC input encoding: 0 -> -1, 1 -> +1
        x = x * 2.0 - 1.0

    y = F.conv2d(x, w, stride=stride, padding=0).squeeze(0)

    if out_bits == 1:
        y = (y > 0).to(torch.int32)

    return y

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("InBits, WeightBits, OutBits, KernelWidth, InChannels, OutChannels, Weights", 
                         [(1, 2,                     1, 2, 1, 1, gen_kernels(2, 1, 1, 2, seed=1234)),
                          (2, 2, output_width(2, 2, 2, 1), 2, 1, 1, gen_kernels(2, 1, 1, 2, seed=1234)),
                          (2, 2, output_width(2, 2, 5, 1), 5, 1, 1, gen_kernels(2, 1, 1, 5, seed=1234)),
                          (2, 3, output_width(2, 3, 3, 1), 3, 1, 1, gen_kernels(3, 1, 1, 3, seed=1234)), # Unsigned data_i
                          (4, 5, output_width(4, 5, 3, 1), 3, 1, 1, gen_kernels(5, 1, 1, 3, seed=1234)),
                          (8, 8, output_width(8, 8, 3, 1), 3, 1, 1, gen_kernels(8, 1, 1, 3, seed=1234)),
                          ])
def test_width(test_name, simulator, InBits, WeightBits, OutBits, KernelWidth, InChannels, OutChannels, Weights):
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    
    # 1. Remove Weights from the CLI parameters dict so cocotb-runner doesn't pass it
    del parameters['Weights'] 
    
    param_str = f"InBits_{InBits}_WeightBits_{WeightBits}_OutBits_{OutBits}_test_{test_name}"
    custom_work_dir = os.path.join(tbpath, "run", "width", param_str, simulator)
    os.makedirs(custom_work_dir, exist_ok=True)
    
    # 2. Calculate total bits to format the Verilog hex string correctly
    total_bits = OutChannels * InChannels * (KernelWidth**2) * WeightBits
    mask = (1 << total_bits) - 1
    packed_weights = Weights & mask

    hex_width = (total_bits + 3) // 4
    vh_path = os.path.join(custom_work_dir, "injected_weights.vh")
    with open(vh_path, "w") as f:
        f.write(
            f"localparam logic signed [{total_bits-1}:0] INJECTED_WEIGHTS = "
            f"{total_bits}'h{packed_weights:0{hex_width}x};\n"
        )

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
@pytest.mark.parametrize("OutBits, KernelWidth, Stride, LineWidthPx, LineCountPx, Weights", 
                         [(output_width(1, 2, 2), 2, 2, 16, 12, gen_kernels(2, 1, 1, 2, seed=1234)),
                          (output_width(1, 2, 4), 4, 4, 16, 12, gen_kernels(2, 1, 1, 4, seed=1234)),
                          (output_width(1, 2, 5), 5, 2, 17, 13, gen_kernels(2, 1, 1, 5, seed=1234))
                          ])

def test_stride(test_name, simulator, OutBits, KernelWidth, Stride, LineWidthPx, LineCountPx, Weights):
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
# Expanded to include InBits, WeightBits, KernelWidth so the DUT scales with the Python math
@pytest.mark.parametrize("InBits, WeightBits, KernelWidth, OutBits, InChannels, OutChannels, Weights", 
                         [(1, 2, 3, 1, 16, 8, gen_kernels(2, 8, 16, 3, seed=1234)),
                          (1, 2, 3, 1, 17, 8, gen_kernels(2, 8, 17, 3, seed=1234)),
                          (1, 2, 3, 1, 8, 8, gen_kernels(2, 8, 8, 3, seed=1234)),
                          (4, 5, 3, output_width(4, 5, 3), 4, 5, gen_kernels(5, 5, 4, 3, seed=1234))
                          ])

def test_channels(test_name, simulator, InBits, WeightBits, KernelWidth, OutBits, InChannels, OutChannels, Weights):
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    del parameters['Weights'] 
    
    param_str = f"InChannels_{InChannels}_OutChannels_{OutChannels}_test_{test_name}"
    custom_work_dir = os.path.join(tbpath, "run", "channels", param_str, simulator)
    os.makedirs(custom_work_dir, exist_ok=True)

    # Calculate total bits using the explicit parameters
    total_bits = OutChannels * InChannels * (KernelWidth**2) * WeightBits
    
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
@pytest.mark.parametrize("LineWidthPx, InBits, OutBits", [("16", "1", output_width(1, 2, 3))])
def test_lint(simulator, LineWidthPx, InBits, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters)

@pytest.mark.parametrize("simulator", ["verilator"])
@pytest.mark.parametrize("LineWidthPx, InBits, OutBits", [("16", "1", output_width(1, 2))])
def test_style(simulator, LineWidthPx, InBits, OutBits):
    # This line must be first
    parameters = dict(locals())
    del parameters['simulator']
    lint(simulator, timescale, tbpath, parameters, compile_args=["--lint-only", "-Wwarn-style", "-Wno-lint"])

class ConvLayerModel():
    def __init__(self, dut, weights: List[List[List[List[int]]]], 
                 output_activation: Optional[List[List[List[int]]]] = None,
                 input_activation: Optional[List[List[List[int]]]] = None):
        
        self._kernel_width = int(dut.KernelWidth.value)
        self._f = np.ones((self._kernel_width,self._kernel_width), dtype=int)
        self._input_activation = input_activation 
        self._output_activation = output_activation

        self._dut = dut
        self._data_o = dut.data_o
        self._data_i = dut.data_i

        self._q = queue.SimpleQueue()

        self._input_width = int(dut.LineWidthPx.value)
        self._input_height = int(dut.LineCountPx.value)
        self._InBits  = int(dut.InBits.value)
        self._OutBits = int(dut.OutBits.value)
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

        for oc in range(self._OutChannels):
            acc = 0
            for ic in range(self._InChannels):
                win = windows[ic]

                if self._InBits == 1:
                    # BNN Mode: 0 -> -1, 1 -> +1
                    win_enc = np.where(win == 1, 1, -1)
                else:
                    # Multi-bit Mode: Ensure Python treats these as signed
                    # If 'win' contains raw bits from the DUT, sign-extend them
                    win_enc = np.vectorize(sign_extend)(win, self._InBits)

                acc += int((self.k[oc, ic] * win_enc).sum())

            if self._OutBits == 1:
                result[oc] = 1 if acc > 0 else 0
            else:
                result[oc] = acc

        return result

    def consume(self):
        """
        The ModelRunner calls this every time it sees a handshake on the INPUT side.
        Because your ModelRunner doesn't know about the 'raw_val' yet, 
        we unpack it here ONE LAST TIME, or update your ModelRunner to handle raw_vals.
        """
        # Get bits from pins
        packed = int(self._dut.data_i.value.integer)
        raw_val = unpack_terms(packed, int(self._dut.InBits.value), self._InChannels)

        # 1. Record input activation if requested
        if self._input_activation is not None:
            y_in = self._enqs // self._input_width
            x_in = self._enqs % self._input_width
            for ic in range(self._InChannels):
                self._input_activation[ic][y_in][x_in] = raw_val[ic]

        # 2. Update sliding windows
        for ic, inp in enumerate(raw_val):
            self._buf[ic] = self.update_window(self._buf[ic], inp)

        # 3. Determine if output is expected
        idx = self._enqs
        x = idx % self._input_width
        y = idx // self._input_width
        self._enqs += 1

        if y >= self._input_height or not self._valid_cycles[y, x]:
            return None

        return self.apply_kernel(self._buf)

    def produce(self, expected):
        assert_resolvable(self._data_o)

        w = int(self._dut.OutBits.value)
        packed = int(self._data_o.value.integer)

        # Write all channels at the SAME (r,c)
        for ch in range(self._OutChannels):
            raw = (packed >> (ch * w)) & ((1 << w) - 1)
            if w == 1:
                got = raw
            else:
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
    
class RandomDataGenerator:
    def __init__(self, dut):
        self._width_p = int(dut.InBits.value)
        self._InChannels = int(dut.InChannels.value)

    def generate(self):
        raw_din = gen_input_channels(self._width_p, self._InChannels)
        packed_din = pack_terms(raw_din, self._width_p)
        return (packed_din, raw_din)

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
    WW = int(dut.WeightBits.value)

    # Number of accepted inputs until first valid output position (x=K-1, y=K-1)
    N_first = (K - 1) * W + (K - 1) + 1

    # We expect exactly ONE output for this test (the first valid position)
    N_out = 1

    rate = 1

    packed_weights = int(os.environ["INJECTED_WEIGHTS_INT"])
    kernels_4d = unpack_kernel_weights(packed_weights, WW, OC, IC, K)
    
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
    WW = int(dut.WeightBits.value)
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
    kernels_4d = unpack_kernel_weights(packed_weights, WW, OC, IC, K)

    model = ConvLayerModel(dut, kernels_4d, output_activation, input_activation)
    m = ModelRunner(dut, model)

    # Consumer fuzzed; producer always drives valid
    om = OutputModel(dut, RateGenerator(dut, out_rate), l_out)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, in_rate), N_in)

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
        ref = torch_conv_ref(input_activation, kernels_4d, S, in_bits=int(dut.InBits.value), out_bits=int(dut.OutBits.value))  # (OC,H_out,W_out)

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
