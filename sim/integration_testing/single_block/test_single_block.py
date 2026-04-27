import numpy as np
import os
from   pathlib import Path
import pytest

from util.utilities  import runner, clock_start_sequence, reset_sequence
from util.bitwise    import unpack_kernel_weights, pack_terms
from util.gen_inputs import gen_kernels, gen_input_channels
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from functional_models.conv_layer import ConvLayerModel, output_width
from functional_models.pool_layer import PoolLayerModel
from util.torch_ref import torch_single_block_ref
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

from util.utilities import assert_resolvable
from util.bitwise import unpack_terms, sign_extend # Adjust imports to your project

class ConvPoolIntegratedModel:
    def __init__(self, dut, conv_params: dict, pool_params: dict):
        self._dut = dut
        
        # 1. Instantiate both models in pure software mode
        self.conv_model = ConvLayerModel(dut=None, **conv_params)
        self.pool_model = PoolLayerModel(dut=None, **pool_params)

        # 2. Store top-level pin widths for unpacking/packing
        self._InBits = int(conv_params["InBits"])
        self._InChannels = int(conv_params["InChannels"])
        
        self._OutBits = int(pool_params["OutBits"])
        self._OutChannels = int(pool_params["OutChannels"])
        
        # 3. State tracking for the final output prints
        self._deqs = 0
        self._OW = self.pool_model._OW  # Final image width comes from the Pool layer

    def consume(self):
        """
        Called by ModelRunner when a valid INPUT handshake occurs on the top-level DUT.
        """
        assert_resolvable(self._dut.data_i)
        packed = int(self._dut.data_i.value.integer)
        
        # Unpack the raw bits from the simulator
        raw_val = unpack_terms(packed, self._InBits, self._InChannels)

        # --- THE PURE PYTHON PIPELINE ---
        
        # 1. Step the Conv Layer
        conv_out = self.conv_model.step(raw_val, in_fire=True)

        # 2. Step the Pool Layer ONLY if Conv produced a valid output
        if conv_out is not None:
            pool_out = self.pool_model.step(conv_out, in_fire=True)
            
            # If Pool also produced an output, we've successfully passed through both layers!
            if pool_out is not None:
                # Wrap in a list so ModelRunner treats it as a single cycle event
                return [pool_out] 

        # If either layer is still warming up, return None
        return None

    def produce(self, expected):
        """
        Called by ModelRunner when a valid OUTPUT handshake occurs on the top-level DUT.
        Checks the final hardware output against the expected pool_out.
        """
        assert_resolvable(self._dut.data_o)
        
        w = self._OutBits
        packed = int(self._dut.data_o.value.integer)

        # Calculate current coordinates for the print statement
        check_idx = self._deqs
        check_r = check_idx // self._OW
        check_c = check_idx % self._OW

        for ch in range(self._OutChannels):
            got_raw = (packed >> (ch * w)) & ((1 << w) - 1)
            
            if w == 1:
                got = got_raw
            else:
                got = sign_extend(got_raw, w)
                
            exp = int(expected[ch])

            print(f"Integrated Output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: expected {exp}, got {got}")

            assert got == exp, (
                f"Mismatch at Integrated Output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: expected {exp}, got {got}"
            )
            
        self._deqs += 1

@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@pytest.mark.parametrize("C_LineWidthPx, C_LineCountPx, C_InBits, C_WeightBits, C_KernelWidth, C_InChannels, C_OutBits, P_KernelWidth, C_Weights, C_OutChannels, P_Mode", 
                         [(12, 8, 2, 2, 3, 1, output_width(2, 2, 3, 1), 2, gen_kernels(2, 1, 1, 3, seed=1234), 1, 0),
                          (12, 8, 4, 4, 3, 4, output_width(4, 4, 3, 4), 2, gen_kernels(4, 8, 4, 3, seed=5678), 8, 0),])
def test_each(test_name, simulator, C_LineWidthPx, C_LineCountPx, C_InBits, C_OutBits, C_KernelWidth, P_KernelWidth, C_WeightBits, C_InChannels, C_Weights, C_OutChannels, P_Mode):
    parameters = dict(locals())
    del parameters['test_name']
    del parameters['simulator']
    
    # 1. Remove Weights from the CLI parameters dict so cocotb-runner doesn't pass it
    del parameters['C_Weights'] 
    
    param_str = f"InBits_{C_InBits}_WeightBits_{C_WeightBits}_OutBits_{C_OutBits}_test_{test_name}"
    custom_work_dir = os.path.join(tbpath, "run", "width", param_str, simulator)
    os.makedirs(custom_work_dir, exist_ok=True)
    
    # 2. Calculate total bits to format the Verilog hex string correctly
    total_bits = C_OutChannels * C_InChannels * (C_KernelWidth**2) * C_WeightBits
    mask = (1 << total_bits) - 1
    packed_weights = C_Weights & mask

    hex_width = (total_bits + 3) // 4
    vh_path = os.path.join(custom_work_dir, "injected_weights.vh")
    with open(vh_path, "w") as f:
        f.write(
            f"localparam logic signed [{total_bits-1}:0] INJECTED_WEIGHTS = "
            f"{total_bits}'h{packed_weights:0{hex_width}x};\n"
        )

    # Pass the massive integer strictly through the OS environment to Cocotb
    os.environ["INJECTED_WEIGHTS_INT"] = str(C_Weights)

    # Define the wrapper path and pass the extra arguments to your runner
    wrapper_path = os.path.join(tbpath, "tb_single_block.sv")

    runner(
        simulator=simulator, 
        timescale=timescale, 
        tbpath=tbpath, 
        params=parameters,
        pymodule="test_single_block",
        testname=test_name, 
        work_dir=custom_work_dir,
        includes=[custom_work_dir],          # Tells simulator where to find injected_weights.vh
        toplevel_override="tb_single_block",   # Forces simulator to use the wrapper as top-level
    )

class RandomDataGenerator:
    def __init__(self, dut):
        self._width_p = int(dut.C_InBits.value)
        self._InChannels = int(dut.C_InChannels.value)

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
    """Drive pixels until the first VALID integrated output position, then expect 1 output."""
    
    # 1. Read DUT parameters
    C_W  = int(dut.C_LineWidthPx.value)
    C_H  = int(dut.C_LineCountPx.value)
    C_K  = int(dut.C_KernelWidth.value)
    C_IC = int(dut.C_InChannels.value)
    C_OC = int(dut.C_OutChannels.value)
    C_WW = int(dut.C_WeightBits.value)
    C_InBits  = int(dut.C_InBits.value)
    C_OutBits = int(dut.C_OutBits.value)
    P_K  = int(dut.P_KernelWidth.value)
    P_M  = int(dut.P_Mode.value)

    N_in = C_W * C_H
    
    # We assume default strides if not exposed as top-level parameters
    C_S = 1 
    P_S = P_K # Standard max pooling stride equals kernel size

    # 2. Setup configuration dictionaries
    packed_weights = int(os.environ["INJECTED_WEIGHTS_INT"])
    kernels_4d = unpack_kernel_weights(packed_weights, C_WW, C_OC, C_IC, C_K)

    conv_params = {
        "KernelWidth": C_K,
        "LineWidthPx": C_W,
        "LineCountPx": C_H,
        "InBits":      C_InBits,
        "OutBits":     C_OutBits,
        "InChannels":  C_IC,
        "OutChannels": C_OC,
        "Stride":      C_S,
        "weights":     kernels_4d
    }

    # Calculate the intermediate dimension (Output of Conv -> Input of Pool)
    P_W = (C_W - C_K) // C_S + 1
    P_H = (C_H - C_K) // C_S + 1

    pool_params = {
        "KernelWidth": P_K,
        "LineWidthPx": P_W,
        "LineCountPx": P_H,
        "InBits":      C_OutBits,
        "OutBits":     C_OutBits, # Pooling generally doesn't change bit width
        "InChannels":  C_OC,
        "OutChannels": C_OC,      # Pooling doesn't change channels
        "Stride":      P_S,
        "PoolMode":    P_M          # Max pooling
    }

    model = ConvPoolIntegratedModel(dut, conv_params, pool_params)
    m = ModelRunner(dut, model)

    om = OutputModel(dut, RateGenerator(dut, 1), 1)                       
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, 1), N_in) 

    dut.ready_i.value = 0
    dut.valid_i.value = 0
    dut.data_i.value = 0

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    tmo_ns = (N_in * 100) + 5000
    try:
        await om.wait(tmo_ns)
    except SimTimeoutError:
        assert False, f"Timed out waiting for first valid output. Expected after ~{N_in} inputs."

    dut.valid_i.value = 0
    dut.ready_i.value = 0

async def rate_tests(dut, in_rate, out_rate):
    # 1. Read DUT parameters
    C_W  = int(dut.C_LineWidthPx.value)
    C_H  = int(dut.C_LineCountPx.value)
    C_K  = int(dut.C_KernelWidth.value)
    C_IC = int(dut.C_InChannels.value)
    C_OC = int(dut.C_OutChannels.value)
    C_WW = int(dut.C_WeightBits.value)
    C_InBits  = int(dut.C_InBits.value)
    C_OutBits = int(dut.C_OutBits.value)
    P_K  = int(dut.P_KernelWidth.value)
    P_M  = int(dut.P_Mode.value)
    C_S = 1 
    P_S = P_K 

    # 2. Calculate dimensions for BOTH layers
    # Conv dimensions
    C_W_out = (C_W - C_K) // C_S + 1
    C_H_out = (C_H - C_K) // C_S + 1
    
    # Pool dimensions (Final Output)
    P_W_out = (C_W_out - P_K) // P_S + 1
    P_H_out = (C_H_out - P_K) // P_S + 1
    
    N_in = C_W * C_H
    l_out = P_W_out * P_H_out   

    input_activation  = [[[0 for _ in range(C_W)] for _ in range(C_H)] for _ in range(C_IC)]
    output_activation = [[[0 for _ in range(P_W_out)] for _ in range(P_H_out)] for _ in range(C_OC)]

    # 3. Setup configuration dictionaries
    packed_weights = int(os.environ["INJECTED_WEIGHTS_INT"])
    kernels_4d = unpack_kernel_weights(packed_weights, C_WW, C_OC, C_IC, C_K)

    conv_params = {
        "KernelWidth": C_K, "LineWidthPx": C_W, "LineCountPx": C_H,
        "InBits": C_InBits, "OutBits": C_OutBits,
        "InChannels": C_IC, "OutChannels": C_OC, "Stride": C_S,
        "weights": kernels_4d,
        "input_activation": input_activation # Let Conv record inputs
    }

    pool_params = {
        "KernelWidth": P_K, "LineWidthPx": C_W_out, "LineCountPx": C_H_out,
        "InBits": C_OutBits, "OutBits": C_OutBits,
        "InChannels": C_OC, "OutChannels": C_OC, "Stride": P_S,
        "PoolMode": P_M,
        "output_activation": output_activation # Let Pool record final outputs
    }

    model = ConvPoolIntegratedModel(dut, conv_params, pool_params)
    m = ModelRunner(dut, model)

    slow = min(in_rate, out_rate)
    slow = max(slow, 0.05) 

    first_out_wait_ns = int((2 * (C_K - 1) * C_W + 2 * (C_K - 1) + 500) / slow)
    timeout_ns        = int((P_H_out * N_in + 1000) / slow)

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

    try:
        await with_timeout(RisingEdge(dut.valid_o), first_out_wait_ns, 'ns')
    except SimTimeoutError:
        assert 0, "Timed out waiting for valid_o high."

    try:
        await om.wait(timeout_ns)
        
        # 4. Verify against PyTorch reference
        final_ref = torch_single_block_ref(input_activation, kernels_4d, C_S, in_bits=C_InBits, out_bits=C_OutBits, mode=P_M, pool_kernel_size=P_K)        

        assert np.allclose(output_activation, final_ref.numpy()), "Output activation does not match..."
        print("Test passed! PyTorch matches integrated model.")

    except SimTimeoutError:
        assert 0, f"Timed out. Expected {l_out} handshakes. Got {om.nproduced()}"

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
