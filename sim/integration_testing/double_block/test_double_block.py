import numpy as np
import os
from   pathlib import Path
import pytest

from util.utilities  import runner, clock_start_sequence, reset_sequence, \
                            sim_verbose, auto_unpack, load_tests_from_csv, inject_weights_and_biases
from util.bitwise    import unpack_kernel_weights, pack_terms, unpack_biases
from util.gen_inputs import gen_kernels, gen_input_channels, gen_biases
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from functional_models.single_block import SingleBlockModel
from functional_models.conv_layer   import output_width
from util.torch_ref import torch_single_block_ref
tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import RisingEdge, FallingEdge, with_timeout
from   cocotb.result import SimTimeoutError
from   cocotb_test import simulator
   
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

class DoubleBlockModel:
    def __init__(self, dut, conv_0_params: dict, pool_0_params: dict, conv_1_params: dict, pool_1_params: dict):
        self._dut = dut
        
        # Pass the dictionaries as the exact arguments the class expects
        self.layer_0_model = SingleBlockModel(dut=None, conv_params=conv_0_params, pool_params=pool_0_params)
        self.layer_1_model = SingleBlockModel(dut=None, conv_params=conv_1_params, pool_params=pool_1_params)

        # 2. Store top-level pin widths for unpacking/packing
        self._InBits = int(conv_0_params["InBits"])
        self._InChannels = int(conv_0_params["InChannels"])
        
        self._OutBits = int(pool_1_params["OutBits"])
        self._OutChannels = int(pool_1_params["OutChannels"])
        
        # 3. State tracking for the final output prints
        self._deqs = 0
        self._OW = self.layer_1_model._OW

    def step(self, raw_val, in_fire=True):
        if not in_fire:
            return None

        # 1. Step Layer 0 (Returns a LIST of tuples, or None)
        l0_outputs = self.layer_0_model.step(raw_val, in_fire=True)

        final_outputs = []

        # 2. Route the burst: Step Layer 1 for EACH valid output from Layer 0
        if l0_outputs is not None:
            for l0_out in l0_outputs:
                
                # Feed the single tuple into Layer 1
                l1_outputs = self.layer_1_model.step(l0_out, in_fire=True)
                
                if l1_outputs is not None:
                    # Layer 1 might also produce a burst, so we extend our final list
                    if isinstance(l1_outputs, list):
                        final_outputs.extend(l1_outputs)
                    else:
                        final_outputs.append(l1_outputs)

        # 3. Return the accumulated burst of final Layer 1 outputs to the ModelRunner
        return final_outputs if len(final_outputs) > 0 else None
    def consume(self):
        """
        Called by ModelRunner when a valid INPUT handshake occurs on the top-level DUT.
        """
        assert_resolvable(self._dut.data_i)
        packed = int(self._dut.data_i.value.integer)
        
        # Unpack the raw bits from the simulator
        raw_val = unpack_terms(packed, self._InBits, self._InChannels)

        # Step entire model
        return self.step(raw_val)

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

            if sim_verbose():
                print(f"Integrated Output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: expected {exp}, got {got}")

            assert got == exp, (
                f"Mismatch at Integrated Output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: expected {exp}, got {got}"
            )
            
        self._deqs += 1

# Format: ("Target_Var", "CSV_Column", lambda <parsed_keys_needed>: func(...))
auto_rules = [
    ("C0_OutBits", "C0_OutBits", lambda C0_InBits, C0_WeightBits, C0_KernelWidth, C0_InChannels, C0_BiasBits: output_width(C0_InBits, C0_WeightBits, C0_KernelWidth, C0_InChannels, C0_BiasBits)),
    ("C1_OutBits", "C1_OutBits", lambda C0_OutBits, C1_WeightBits, C1_KernelWidth, C0_OutChannels, C1_BiasBits: output_width(C0_OutBits, C1_WeightBits, C1_KernelWidth, C0_OutChannels, C1_BiasBits))
]

# Format: ("Target_Var", lambda <parsed_keys_needed>: func(...))
gen_rules = [
    ("C0_Weights", lambda C0_WeightBits, C0_OutChannels, C0_InChannels, C0_KernelWidth: gen_kernels(C0_WeightBits, C0_OutChannels, C0_InChannels, C0_KernelWidth, seed=1234)),
    ("C1_Weights", lambda C1_WeightBits, C1_OutChannels, C0_OutChannels, C1_KernelWidth: gen_kernels(C1_WeightBits, C1_OutChannels, C0_OutChannels, C1_KernelWidth, seed=1234)),
    ("C0_Biases",  lambda C0_BiasBits, C0_OutChannels: gen_biases(C0_BiasBits, C0_OutChannels, seed=1234)),
    ("C1_Biases",  lambda C1_BiasBits, C1_OutChannels: gen_biases(C1_BiasBits, C1_OutChannels, seed=1234))
]

# Generate the list of tuples dynamically from the CSV
TEST_CASES = load_tests_from_csv(os.path.join(tbpath, "test_cases.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
# Pass the dynamically generated list directly into parametrize!
@auto_unpack(TEST_CASES)
def test_each(test_name, simulator, 
              C0_LineWidthPx, C0_LineCountPx, C0_InBits, C0_OutBits,
              C0_KernelWidth, C0_WeightBits, C0_BiasBits, C0_InChannels, C0_OutChannels,
              C0_Stride, C0_Padding, C0_Weights, C0_Biases,  P0_KernelWidth, P0_Mode, 
              C1_OutBits, C1_KernelWidth, C1_WeightBits, C1_BiasBits, C1_OutChannels, 
              C1_Stride, C1_Padding, C1_Weights, C1_Biases, P1_KernelWidth, P1_Mode):
    parameters = dict(locals())
    parameters.pop('C0_Weights', None)
    parameters.pop('C0_Biases', None)
    parameters.pop('C1_Weights', None)
    parameters.pop('C1_Biases', None)
    param_str = f"InBits_{C0_InBits}_OutBits0_{C0_OutBits}_OutBits1_{C1_OutBits}_test_{test_name}"
    
    weight_bits_0 = C0_OutChannels * C0_InChannels * (C0_KernelWidth**2) * C0_WeightBits
    weight_bits_1 = C1_OutChannels * C0_OutChannels * (C1_KernelWidth**2) * C1_WeightBits

    bias_bits_0 = C0_OutChannels * C0_BiasBits
    bias_bits_1 = C1_OutChannels * C1_BiasBits
    
    custom_work_dir = inject_weights_and_biases(
        simulator=simulator, parameters=parameters, param_str=param_str, 
        tbpath=tbpath, test_class="each", Weights=C0_Weights, Biases=C0_Biases, 
        weight_bits=weight_bits_0, bias_bits=bias_bits_0, layer=0)
    
    inject_weights_and_biases(
        simulator=simulator, parameters=parameters, param_str=param_str, 
        tbpath=tbpath, test_class="each", Weights=C1_Weights, Biases=C1_Biases, 
        weight_bits=weight_bits_1, bias_bits=bias_bits_1, layer=1)

    runner(
        simulator=simulator, timescale=timescale, tbpath=tbpath, params=parameters,
        pymodule="test_double_block", testname=test_name, work_dir=custom_work_dir,
        sim_build=custom_work_dir, includes=[custom_work_dir], toplevel_override="tb_double_block", 
    )

class RandomDataGenerator:
    def __init__(self, dut):
        self._width_p = int(dut.C0_InBits.value)
        self._InChannels = int(dut.C0_InChannels.value)

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
    """Drive pixels until the first VALID integrated output position, then expect 1 output."""
    
    # 1. Read DUT parameters for Block 0
    C0_W  = int(dut.C0_LineWidthPx.value)
    C0_H  = int(dut.C0_LineCountPx.value)
    C0_K  = int(dut.C0_KernelWidth.value)
    C0_IC = int(dut.C0_InChannels.value)
    C0_OC = int(dut.C0_OutChannels.value)
    C0_WW = int(dut.C0_WeightBits.value)
    C0_BB = int(dut.C0_BiasBits.value)
    C0_InBits  = int(dut.C0_InBits.value)
    C0_OutBits = int(dut.C0_OutBits.value)
    C0_P  = int(dut.C0_Padding.value)
    C0_S  = 1 
    
    P0_K  = int(dut.P0_KernelWidth.value)
    P0_M  = int(dut.P0_Mode.value)
    P0_S  = P0_K 

    # Read DUT parameters for Block 1 (Deriving linked params from Block 0)
    C1_K  = int(dut.C1_KernelWidth.value)
    C1_IC = C0_OC
    C1_OC = int(dut.C1_OutChannels.value)
    C1_WW = int(dut.C1_WeightBits.value)
    C1_BB = int(dut.C1_BiasBits.value)
    C1_InBits  = C0_OutBits
    C1_OutBits = int(dut.C1_OutBits.value)
    C1_P  = int(dut.C1_Padding.value)
    C1_S  = 1
    
    P1_K  = int(dut.P1_KernelWidth.value)
    P1_M  = int(dut.P1_Mode.value)
    P1_S  = P1_K

    N_in = C0_W * C0_H

    # 2. Calculate dimensions for ALL stages
    # Block 0
    C0_W_out = ((C0_W + 2 * C0_P - C0_K) // C0_S) + 1
    C0_H_out = ((C0_H + 2 * C0_P - C0_K) // C0_S) + 1
    P0_W_out = ((C0_W_out - P0_K) // P0_S) + 1
    P0_H_out = ((C0_H_out - P0_K) // P0_S) + 1

    # Block 1
    C1_W_out = ((P0_W_out + 2 * C1_P - C1_K) // C1_S) + 1
    C1_H_out = ((P0_H_out + 2 * C1_P - C1_K) // C1_S) + 1
    P1_W_out = ((C1_W_out - P1_K) // P1_S) + 1
    P1_H_out = ((C1_H_out - P1_K) // P1_S) + 1

    # 3. Unpack Weights for both blocks
    packed_weights_0 = int(os.environ["INJECTED_WEIGHTS_0_INT"])
    kernels_4d_0 = unpack_kernel_weights(packed_weights_0, C0_WW, C0_OC, C0_IC, C0_K)

    packed_weights_1 = int(os.environ["INJECTED_WEIGHTS_1_INT"])
    kernels_4d_1 = unpack_kernel_weights(packed_weights_1, C1_WW, C1_OC, C1_IC, C1_K)

    packed_biases_0 = int(os.environ["INJECTED_BIASES_0_INT"])
    biases_0 = unpack_biases(packed_biases_0, C0_BB, C0_OC)

    packed_biases_1 = int(os.environ["INJECTED_BIASES_1_INT"])
    biases_1 = unpack_biases(packed_biases_1, C1_BB, C1_OC)

    # 4. Setup configuration dictionaries
    conv_0_params = {
        "KernelWidth": C0_K, "LineWidthPx": C0_W, "LineCountPx": C0_H,
        "InBits": C0_InBits, "OutBits": C0_OutBits, "WeightBits": C0_WW, "BiasBits": C0_BB,
        "InChannels": C0_IC, "OutChannels": C0_OC, "Stride": C0_S,
        "Padding": C0_P, "weights": kernels_4d_0, "biases": biases_0
    }
    pool_0_params = {
        "KernelWidth": P0_K, "LineWidthPx": C0_W_out, "LineCountPx": C0_H_out,
        "InBits": C0_OutBits, "OutBits": C0_OutBits,
        "InChannels": C0_OC, "OutChannels": C0_OC, "Stride": P0_S,
        "PoolMode": P0_M
    }
    conv_1_params = {
        "KernelWidth": C1_K, "LineWidthPx": P0_W_out, "LineCountPx": P0_H_out,
        "InBits": C1_InBits, "OutBits": C1_OutBits, "WeightBits": C1_WW, "BiasBits": C1_BB,
        "InChannels": C1_IC, "OutChannels": C1_OC, "Stride": C1_S,
        "Padding": C1_P, "weights": kernels_4d_1, "biases": biases_1
    }
    pool_1_params = {
        "KernelWidth": P1_K, "LineWidthPx": C1_W_out, "LineCountPx": C1_H_out,
        "InBits": C1_OutBits, "OutBits": C1_OutBits,
        "InChannels": C1_OC, "OutChannels": C1_OC, "Stride": P1_S,
        "PoolMode": P1_M
    }

    # Instantiate Double Block Model
    model = DoubleBlockModel(dut, conv_0_params, pool_0_params, conv_1_params, pool_1_params)
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
    # 1. Read DUT parameters (Same as single_test)
    C0_W  = int(dut.C0_LineWidthPx.value)
    C0_H  = int(dut.C0_LineCountPx.value)
    C0_K  = int(dut.C0_KernelWidth.value)
    C0_IC = int(dut.C0_InChannels.value)
    C0_OC = int(dut.C0_OutChannels.value)
    C0_WW = int(dut.C0_WeightBits.value)
    C0_BB = int(dut.C0_BiasBits.value)
    C0_InBits  = int(dut.C0_InBits.value)
    C0_OutBits = int(dut.C0_OutBits.value)
    C0_P  = int(dut.C0_Padding.value)
    C0_S  = 1 
    
    P0_K  = int(dut.P0_KernelWidth.value)
    P0_M  = int(dut.P0_Mode.value)
    P0_S  = P0_K 

    C1_K  = int(dut.C1_KernelWidth.value)
    C1_IC = C0_OC
    C1_OC = int(dut.C1_OutChannels.value)
    C1_WW = int(dut.C1_WeightBits.value)
    C1_BB = int(dut.C1_BiasBits.value)
    C1_InBits  = C0_OutBits
    C1_OutBits = int(dut.C1_OutBits.value)
    C1_P  = int(dut.C1_Padding.value)
    C1_S  = 1
    
    P1_K  = int(dut.P1_KernelWidth.value)
    P1_M  = int(dut.P1_Mode.value)
    P1_S  = P1_K

    # 2. Calculate dimensions for ALL stages
    C0_W_out = ((C0_W + 2 * C0_P - C0_K) // C0_S) + 1
    C0_H_out = ((C0_H + 2 * C0_P - C0_K) // C0_S) + 1
    P0_W_out = ((C0_W_out - P0_K) // P0_S) + 1
    P0_H_out = ((C0_H_out - P0_K) // P0_S) + 1

    C1_W_out = ((P0_W_out + 2 * C1_P - C1_K) // C1_S) + 1
    C1_H_out = ((P0_H_out + 2 * C1_P - C1_K) // C1_S) + 1
    P1_W_out = ((C1_W_out - P1_K) // P1_S) + 1
    P1_H_out = ((C1_H_out - P1_K) // P1_S) + 1
    
    N_in = C0_W * C0_H
    l_out = P1_W_out * P1_H_out   

    # Final output shapes are based on Block 1 Pool dimensions
    input_activation  = [[[0 for _ in range(C0_W)] for _ in range(C0_H)] for _ in range(C0_IC)]
    output_activation = [[[0 for _ in range(P1_W_out)] for _ in range(P1_H_out)] for _ in range(C1_OC)]

    # 3. Unpack Weights for both blocks
    packed_weights_0 = int(os.environ["INJECTED_WEIGHTS_0_INT"])
    kernels_4d_0 = unpack_kernel_weights(packed_weights_0, C0_WW, C0_OC, C0_IC, C0_K)

    packed_weights_1 = int(os.environ["INJECTED_WEIGHTS_1_INT"])
    kernels_4d_1 = unpack_kernel_weights(packed_weights_1, C1_WW, C1_OC, C1_IC, C1_K)

    packed_biases_0 = int(os.environ["INJECTED_BIASES_0_INT"])
    biases_0 = unpack_biases(packed_biases_0, C0_BB, C0_OC)

    packed_biases_1 = int(os.environ["INJECTED_BIASES_1_INT"])
    biases_1 = unpack_biases(packed_biases_1, C1_BB, C1_OC)

    # 4. Setup configuration dictionaries
    conv_0_params = {
        "KernelWidth": C0_K, "LineWidthPx": C0_W, "LineCountPx": C0_H,
        "InBits": C0_InBits, "OutBits": C0_OutBits, "WeightBits": C0_WW,
        "BiasBits": C0_BB, "InChannels": C0_IC, "OutChannels": C0_OC, "Stride": C0_S,
        "Padding": C0_P, "weights": kernels_4d_0, "biases": biases_0,
        "input_activation": input_activation # Let Conv0 record initial inputs
    }
    pool_0_params = {
        "KernelWidth": P0_K, "LineWidthPx": C0_W_out, "LineCountPx": C0_H_out,
        "InBits": C0_OutBits, "OutBits": C0_OutBits,
        "InChannels": C0_OC, "OutChannels": C0_OC, "Stride": P0_S,
        "PoolMode": P0_M
    }
    conv_1_params = {
        "KernelWidth": C1_K, "LineWidthPx": P0_W_out, "LineCountPx": P0_H_out,
        "InBits": C1_InBits, "OutBits": C1_OutBits, "WeightBits": C1_WW,
        "InChannels": C1_IC, "OutChannels": C1_OC, "Stride": C1_S,
        "Padding": C1_P, "weights": kernels_4d_1, "biases": biases_1, "BiasBits": C1_BB
    }
    pool_1_params = {
        "KernelWidth": P1_K, "LineWidthPx": C1_W_out, "LineCountPx": C1_H_out,
        "InBits": C1_OutBits, "OutBits": C1_OutBits,
        "InChannels": C1_OC, "OutChannels": C1_OC, "Stride": P1_S,
        "PoolMode": P1_M,
        "output_activation": output_activation # Let Pool1 record final outputs
    }

    model = DoubleBlockModel(dut, conv_0_params, pool_0_params, conv_1_params, pool_1_params)
    m = ModelRunner(dut, model)

    slow = min(in_rate, out_rate)
    slow = max(slow, 0.05) 

    first_out_wait_ns = int((2 * (C0_K - 1) * C0_W + 2 * (C0_K - 1) + 1000) / slow)
    timeout_ns        = int((P1_H_out * N_in + 2000) / slow)

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
        
        # 5. Verify against PyTorch reference
        # We can simply chain your existing single_block_ref function!
        ref_block_0 = torch_single_block_ref(
            input_activation, kernels_4d_0, C0_S, 
            in_bits=C0_InBits, out_bits=C0_OutBits, 
            mode=P0_M, pool_kernel_size=P0_K, padding=C0_P, biases=biases_0
        )
        
        final_ref = torch_single_block_ref(
            ref_block_0, kernels_4d_1, C1_S, 
            in_bits=C1_InBits, out_bits=C1_OutBits, 
            mode=P1_M, pool_kernel_size=P1_K, padding=C1_P, biases=biases_1
        )

        assert np.allclose(output_activation, final_ref.numpy()), "Output activation does not match..."
        if sim_verbose():
            print("Test passed! PyTorch matches double integrated model.")

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
