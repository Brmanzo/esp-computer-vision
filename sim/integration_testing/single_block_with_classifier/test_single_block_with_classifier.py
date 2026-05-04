import numpy as np
import os
from   pathlib import Path
import pytest

from util.utilities  import runner, clock_start_sequence, reset_sequence, \
                            sim_verbose, auto_unpack, load_tests_from_csv, inject_weights_and_biases
from util.bitwise    import unpack_kernel_weights, unpack_weights, pack_terms, unpack_biases, sign_extend, unpack_terms
from util.gen_inputs import gen_kernels, gen_input_channels, gen_biases
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from functional_models.single_block import SingleBlockModel
from functional_models.conv_layer   import ConvLayerModel, output_width
from functional_models.classifier_layer import ClassifierLayerModel
from util.utilities import assert_resolvable

tbpath = Path(__file__).parent

import cocotb
from   cocotb.triggers import RisingEdge, FallingEdge, with_timeout
from   cocotb.result import SimTimeoutError
from   cocotb_test import simulator
   
import random
random.seed(50)

timescale = "1ps/1ps"
tests = ['reset_test'
        ,'single_test'
        ,'inout_fuzz_test'
        ,'in_fuzz_test'
        ,'out_fuzz_test'
        ,'full_bw_test']

class SingleBlockWithClassifierModel:
    def __init__(self, dut, conv_0_params: dict, pool_0_params: dict, conv_1_params: dict, classifier_params: dict):
        self._dut = dut
        
        # Pass the dictionaries as the exact arguments the class expects
        self.layer_0_model = SingleBlockModel(dut=None, conv_params=conv_0_params, pool_params=pool_0_params)
        self.layer_1_model = ConvLayerModel(dut=None, **conv_1_params)
        self.layer_2_model = ClassifierLayerModel(dut=None, **classifier_params)

        # 2. Store top-level pin widths for unpacking/packing
        self._InBits = int(conv_0_params["InBits"])
        self._InChannels = int(conv_0_params["InChannels"])
        
        self._OutBits = int(classifier_params["bus_bits"])
        self._OutChannels = 1 # Classifier produces a single Class ID
        
        # 3. State tracking for the final output prints
        self._deqs = 0

    def step(self, raw_val, in_fire=True):
        if not in_fire:
            return None

        # 1. Step Layer 0 (Conv+Pool)
        l0_outputs = self.layer_0_model.step(raw_val, in_fire=True)

        final_outputs = []

        # 2. Step Layer 1 (Conv)
        if l0_outputs is not None:
            for l0_out in l0_outputs:
                l1_outputs = self.layer_1_model.step(l0_out, in_fire=True)
                
                if l1_outputs is not None:
                    # 3. Step Layer 2 (Classifier)
                    for l1_out in l1_outputs:
                        # Convert tuple to list for the classifier's step method
                        l2_out = self.layer_2_model.step(list(l1_out))
                        if l2_out is not None:
                            # Classifier returns [(class_id, logits)]
                            # We only care about the class_id for the output handshake
                            final_outputs.append(l2_out[0][0])

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
        Checks the final hardware output against the expected class ID.
        """
        assert_resolvable(self._dut.data_o)
        got_id = int(self._dut.data_o.value.integer)
        expected_id = int(expected)

        if sim_verbose():
            print(f"Integrated Output #{self._deqs}: expected Class {expected_id}, got {got_id}")

        assert got_id == expected_id, (
            f"Mismatch at Integrated Output #{self._deqs}: expected {expected_id}, got {got_id}"
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
    ("C1_Biases",  lambda C1_BiasBits, C1_OutChannels: gen_biases(C1_BiasBits, C1_OutChannels, seed=1234)),
    ("C2_Weights", lambda ClassWeightBits, ClassCount, C1_OutChannels: gen_kernels(ClassWeightBits, ClassCount, C1_OutChannels, 1, seed=1234)),
    ("C2_Biases",  lambda ClassBiasBits, ClassCount: gen_biases(ClassBiasBits, ClassCount, seed=1234))
]

# Generate the list of tuples dynamically from the CSV
TEST_CASES = load_tests_from_csv(os.path.join(tbpath, "test_cases.csv"), auto_rules, gen_rules)
@pytest.mark.parametrize("test_name", tests)
@pytest.mark.parametrize("simulator", ["verilator", "icarus"])
@auto_unpack(TEST_CASES)
def test_each(test_name, simulator, 
              C0_LineWidthPx, C0_LineCountPx, C0_InBits, C0_OutBits,
              C0_KernelWidth, C0_WeightBits, C0_BiasBits, C0_InChannels, C0_OutChannels,
              C0_Stride, C0_Padding, C0_Weights, C0_Biases,  P0_KernelWidth, P0_Mode, 
              C1_OutBits, C1_KernelWidth, C1_WeightBits, C1_BiasBits, C1_OutChannels, 
              C1_Stride, C1_Padding, C1_Weights, C1_Biases,
              ClassCount, BusBits, ClassWeightBits, ClassBiasBits, C2_Weights, C2_Biases):
    parameters = dict(locals())
    parameters.pop('C0_Weights', None)
    parameters.pop('C0_Biases', None)
    parameters.pop('C1_Weights', None)
    parameters.pop('C1_Biases', None)
    parameters.pop('C2_Weights', None)
    parameters.pop('C2_Biases', None)
    param_str = f"InBits_{C0_InBits}_OutBits0_{C0_OutBits}_OutBits1_{C1_OutBits}_test_{test_name}"
    
    weight_bits_0 = C0_OutChannels * C0_InChannels * (C0_KernelWidth**2) * C0_WeightBits
    weight_bits_1 = C1_OutChannels * C0_OutChannels * (C1_KernelWidth**2) * C1_WeightBits
    weight_bits_2 = ClassCount * C1_OutChannels * ClassWeightBits

    bias_bits_0 = C0_OutChannels * C0_BiasBits
    bias_bits_1 = C1_OutChannels * C1_BiasBits
    bias_bits_2 = ClassCount * ClassBiasBits
    
    custom_work_dir = inject_weights_and_biases(
        simulator=simulator, parameters=parameters, param_str=param_str, 
        tbpath=tbpath, test_class="each", Weights=C0_Weights, Biases=C0_Biases, 
        weight_bits=weight_bits_0, bias_bits=bias_bits_0, layer=0)
    
    inject_weights_and_biases(
        simulator=simulator, parameters=parameters, param_str=param_str, 
        tbpath=tbpath, test_class="each", Weights=C1_Weights, Biases=C1_Biases, 
        weight_bits=weight_bits_1, bias_bits=bias_bits_1, layer=1)

    inject_weights_and_biases(
        simulator=simulator, parameters=parameters, param_str=param_str, 
        tbpath=tbpath, test_class="each", Weights=C2_Weights, Biases=C2_Biases, 
        weight_bits=weight_bits_2, bias_bits=bias_bits_2, layer=2)

    runner(
        simulator=simulator, timescale=timescale, tbpath=tbpath, params=parameters,
        pymodule="test_single_block_with_classifier", testname=test_name, work_dir=custom_work_dir,
        sim_build=custom_work_dir, includes=[custom_work_dir], toplevel_override="tb_single_block_with_classifier", 
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
    
    # 1. Read DUT parameters
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
    C0_S  = int(dut.C0_Stride.value)
    
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
    C1_S  = int(dut.C1_Stride.value)

    CL_TC = int(dut.ClassifierTermCount.value)
    CL_CC = int(dut.ClassCount.value)
    CL_BB = int(dut.BusBits.value)
    CL_WB = int(dut.ClassWeightBits.value)
    CL_BI = int(dut.ClassBiasBits.value)

    N_in = C0_W * C0_H

    # 2. Calculate dimensions
    C0_W_out = ((C0_W + 2 * C0_P - C0_K) // C0_S) + 1
    C0_H_out = ((C0_H + 2 * C0_P - C0_K) // C0_S) + 1
    P0_W_out = ((C0_W_out - P0_K) // P0_S) + 1
    P0_H_out = ((C0_H_out - P0_K) // P0_S) + 1

    # 3. Unpack Weights
    kernels_4d_0 = unpack_kernel_weights(int(os.environ["INJECTED_WEIGHTS_0_INT"]), C0_WW, C0_OC, C0_IC, C0_K)
    biases_0 = unpack_biases(int(os.environ["INJECTED_BIASES_0_INT"]), C0_BB, C0_OC)

    kernels_4d_1 = unpack_kernel_weights(int(os.environ["INJECTED_WEIGHTS_1_INT"]), C1_WW, C1_OC, C1_IC, C1_K)
    biases_1 = unpack_biases(int(os.environ["INJECTED_BIASES_1_INT"]), C1_BB, C1_OC)

    weights_2 = unpack_weights(int(os.environ["INJECTED_WEIGHTS_2_INT"]), CL_WB, CL_CC, C1_OC)
    biases_2 = unpack_biases(int(os.environ["INJECTED_BIASES_2_INT"]), CL_BI, CL_CC)

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
        "InBits": C0_OutBits, "OutBits": C1_OutBits, "WeightBits": C1_WW, "BiasBits": C1_BB,
        "InChannels": C1_IC, "OutChannels": C1_OC, "Stride": C1_S,
        "Padding": C1_P, "weights": kernels_4d_1, "biases": biases_1
    }
    classifier_params = {
        "term_bits": C1_OutBits, "term_count": CL_TC, "bus_bits": CL_BB,
        "in_channels": C1_OC, "class_count": CL_CC, "weights": weights_2, "biases": biases_2
    }

    model = SingleBlockWithClassifierModel(dut, conv_0_params, pool_0_params, conv_1_params, classifier_params)
    m = ModelRunner(dut, model)

    om = OutputModel(dut, RateGenerator(dut, 1), 1)                       
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, 1), N_in)

    timeout_ns = N_in * 2000

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)

    im.start()
    om.start()

    try:
        await om.wait(timeout_ns)
        print("Success: Final Classification output produced!")
    except SimTimeoutError:
        assert 0, f"Timed out. Expected 1 handshake. Got {om.nproduced()}"

async def rate_tests(dut, in_rate, out_rate):
    # 1. Read DUT parameters
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
    C0_S  = int(dut.C0_Stride.value)
    
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
    C1_S  = int(dut.C1_Stride.value)

    CL_TC = int(dut.ClassifierTermCount.value)
    CL_CC = int(dut.ClassCount.value)
    CL_BB = int(dut.BusBits.value)
    CL_WB = int(dut.ClassWeightBits.value)
    CL_BI = int(dut.ClassBiasBits.value)

    N_in = C0_W * C0_H

    # 2. Calculate dimensions
    C0_W_out = ((C0_W + 2 * C0_P - C0_K) // C0_S) + 1
    C0_H_out = ((C0_H + 2 * C0_P - C0_K) // C0_S) + 1
    P0_W_out = ((C0_W_out - P0_K) // P0_S) + 1
    P0_H_out = ((C0_H_out - P0_K) // P0_S) + 1

    # 3. Unpack Weights
    kernels_4d_0 = unpack_kernel_weights(int(os.environ["INJECTED_WEIGHTS_0_INT"]), C0_WW, C0_OC, C0_IC, C0_K)
    biases_0 = unpack_biases(int(os.environ["INJECTED_BIASES_0_INT"]), C0_BB, C0_OC)

    kernels_4d_1 = unpack_kernel_weights(int(os.environ["INJECTED_WEIGHTS_1_INT"]), C1_WW, C1_OC, C1_IC, C1_K)
    biases_1 = unpack_biases(int(os.environ["INJECTED_BIASES_1_INT"]), C1_BB, C1_OC)

    weights_2 = unpack_weights(int(os.environ["INJECTED_WEIGHTS_2_INT"]), CL_WB, CL_CC, C1_OC)
    biases_2 = unpack_biases(int(os.environ["INJECTED_BIASES_2_INT"]), CL_BI, CL_CC)

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
        "InBits": C0_OutBits, "OutBits": C1_OutBits, "WeightBits": C1_WW, "BiasBits": C1_BB,
        "InChannels": C1_IC, "OutChannels": C1_OC, "Stride": C1_S,
        "Padding": C1_P, "weights": kernels_4d_1, "biases": biases_1
    }
    classifier_params = {
        "term_bits": C1_OutBits, "term_count": CL_TC, "bus_bits": CL_BB,
        "in_channels": C1_OC, "class_count": CL_CC, "weights": weights_2, "biases": biases_2
    }

    model = SingleBlockWithClassifierModel(dut, conv_0_params, pool_0_params, conv_1_params, classifier_params)
    m = ModelRunner(dut, model)

    slow = min(in_rate, out_rate)
    timeout_ns = int(N_in * 2000 / slow)

    om = OutputModel(dut, RateGenerator(dut, out_rate), 1)
    im = InputModel(dut, RandomDataGenerator(dut), RateGenerator(dut, in_rate), N_in)

    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)

    im.start()
    om.start()

    try:
        await om.wait(timeout_ns)
        print("Success: Final Classification output produced!")
    except SimTimeoutError:
        assert 0, f"Timed out. Expected 1 handshake. Got {om.nproduced()}"

@cocotb.test
async def inout_fuzz_test(dut):
    """Test both Input and Output fuzzing"""
    await rate_tests(dut, random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))

@cocotb.test
async def in_fuzz_test(dut):
    """Test Input fuzzing"""
    await rate_tests(dut, random.uniform(0.1, 0.9), 1.0)

@cocotb.test
async def out_fuzz_test(dut):
    """Test Output fuzzing"""
    await rate_tests(dut, 1.0, random.uniform(0.1, 0.9))

@cocotb.test
async def full_bw_test(dut):
    """Test Full Bandwidth"""
    await rate_tests(dut, 1.0, 1.0)
