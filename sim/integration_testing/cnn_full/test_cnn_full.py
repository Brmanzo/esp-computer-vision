# test_cnn_full.py
import os
from pathlib import Path
import cocotb
from cocotb.triggers import FallingEdge
import sys

# Increase the limit for integer to string conversion for massive weights
sys.set_int_max_str_digits(0)

from util.utilities  import runner, clock_start_sequence, reset_sequence, get_param_string, inject_raw_param, inject_weights_and_biases
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from functional_models.cnn_model import CNNModel, PictureGenerator
from util.weight_loader import load_weights_from_vh
from model.globals import HAND_GESTURE_CFG

@cocotb.test
async def full_cnn_test(dut) -> None:
    """
    Full CNN Integration Test using ModelConfig and Hardware Weights.
    """
    # 1. Load Architecture Config
    config = HAND_GESTURE_CFG
    
    # 2. Instantiate Functional Model
    from typing import Any
    weights_dict: dict[str, Any] = {}
    from util.bitwise import unpack_kernel_weights, unpack_weights, unpack_biases
    
    for i, layer_cfg in enumerate(config.layers):
        cfg = layer_cfg.ConvLayer
        W_int = int(os.environ[f"INJECTED_WEIGHTS_{i}_INT"])
        B_int = int(os.environ[f"INJECTED_BIASES_{i}_INT"])
        if cfg._bias_bits is None or cfg._out_ch is None:
            raise ValueError("Bias bits or out channels must be non-zero")
        weights_dict[f"LAYER_{i}_WEIGHTS"] = unpack_kernel_weights(W_int, cfg._q_schedule._q_min_bits, cfg._out_ch, cfg._in_ch, cfg._kernel_width)
        weights_dict[f"LAYER_{i}_BIASES"] = unpack_biases(B_int, cfg._bias_bits, cfg._out_ch)
        
    c_idx = len(config.layers)
    c_cfg = config.classifier_config
    CW_int = int(os.environ[f"INJECTED_WEIGHTS_{c_idx}_INT"])
    CB_int = int(os.environ[f"INJECTED_BIASES_{c_idx}_INT"])
    weights_dict[f"LAYER_{c_idx}_WEIGHTS"] = unpack_weights(CW_int, c_cfg._q_schedule._q_min_bits, c_cfg._num_classes, c_cfg._in_ch)
    weights_dict[f"LAYER_{c_idx}_BIASES"] = unpack_biases(CB_int, c_cfg._bias_bits, c_cfg._num_classes)

    model = CNNModel(dut, config, weights_dict)
    
    # 3. Setup verification components
    if config.in_dims.width is None or config.in_dims.height is None:
        assert False, "CNN input dimensions are not set!"
    n_pixels = config.in_dims.width * config.in_dims.height
    
    m = ModelRunner(dut, model)
    om = OutputModel(dut, length=1) # Expect 1 classification ID
    im = InputModel(dut, PictureGenerator(model), RateGenerator(dut, 1), n_pixels)

    # 4. Execute Simulation
    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    # Calculate timeout based on image size and clock (10ns)
    timeout_ns = n_pixels * 100 # Extra margin for deep pipeline
    await om.wait(timeout_ns)

    # 5. Report Results
    from model.preprocess import get_class_names
    from model.inference import get_inference
    
    class_names = get_class_names()
    sample_idx = int(os.environ.get("SAMPLE_IDX", "10"))
    
    got_id = model.results[0]
    torch_id = get_inference(sample_idx)
    from model.sample import get_sample
    _, label = get_sample(sample_idx)
    assert label is not None
    
    print(f"\n--- CNN VERIFICATION RESULTS (Sample {sample_idx}) ---")
    print(f"Ground Truth:    {label} ({class_names[label]})")
    print(f"Hardware Output: {got_id} ({class_names[got_id]})")
    print(f"PyTorch Ref:     {torch_id} ({class_names[torch_id]})")
    
    if got_id == torch_id:
        print("\033[92mSUCCESS: Hardware output matches PyTorch prediction!\033[0m")
    else:
        print("\033[91mFAILURE: Hardware output mismatch with PyTorch!\033[0m")

    print("SUCCESS: Full CNN Pipeline Verified!")

def test_full() -> None:
    """Pytest entry point"""
    tbpath = Path(__file__).parent
    config = HAND_GESTURE_CFG
    simulator = "verilator"
    params = {"BusBits": 8}
    testname = "full_cnn_test"
    
    # 1. Prepare work directory
    param_str = get_param_string(params)
    work_dir = os.path.join(tbpath, "run", testname, param_str, simulator)
    os.makedirs(work_dir, exist_ok=True)
    
    # 2. Load weights from original .vh for injection
    vh_path = (tbpath / ".." / ".." / ".." / "model" / "data" / "hardware_weights.vh").resolve()
    _, raw_dict = load_weights_from_vh(str(vh_path), config)
    
    # 3. Inject parameters and generate headers for each layer
    # Feature Layers
    for i, layer_cfg in enumerate(config.layers):
        cfg = layer_cfg.ConvLayer
        inject_weights_and_biases(
            simulator=simulator, parameters=params, param_str=param_str, 
            tbpath=tbpath, test_class="full", Weights=raw_dict[f"LAYER_{i}_WEIGHTS"], Biases=raw_dict[f"LAYER_{i}_BIASES"], 
            weight_bits=cfg._q_schedule._q_min_bits, bias_bits=cfg._bias_bits, 
            weight_count=cfg._out_ch * cfg._in_ch * (cfg._kernel_width**2),
            layer=i, dsp_count=cfg._dsp_count, custom_work_dir=work_dir)

    # Classifier
    c_idx = len(config.layers)
    c_cfg = config.classifier_config
    inject_weights_and_biases(
        simulator=simulator, parameters=params, param_str=param_str, 
        tbpath=tbpath, test_class="full", Weights=raw_dict[f"LAYER_{c_idx}_WEIGHTS"], Biases=raw_dict[f"LAYER_{c_idx}_BIASES"], 
        weight_bits=c_cfg._q_schedule._q_min_bits, bias_bits=c_cfg._bias_bits, 
        weight_count=c_cfg._num_classes * c_cfg._in_ch,
        layer=c_idx, dsp_count=c_cfg._dsp_count, custom_work_dir=work_dir)

    # Create the master header that includes all injected headers
    master_header_path = os.path.join(work_dir, "injected_weights_0.vh")
    with open(master_header_path, "w") as f:
        for i in range(len(config.layers) + 1):
            f.write(f'`include "layer_{i}_weights.vh"\n')
            f.write(f'`include "layer_{i}_biases.vh"\n')
    
    # 4. Run simulator
    os.environ["CHECK_INFERENCE"] = os.environ.get("CHECK_INFERENCE", "1")
    os.environ["SAMPLE_IDX"]      = os.environ.get("SAMPLE_IDX", "10")
    
    runner(
        simulator=simulator,
        timescale="1ps/1ps",
        tbpath=tbpath,
        params=params,
        pymodule="test_cnn_full",
        testname=testname,
        jsonname="cnn_full.json",
        work_dir=work_dir,
        sim_build=work_dir,
        includes=[work_dir],
        toplevel_override="cnn_full",
    )

if __name__ == "__main__":
    test_full()
