# test_cnn_full.py
import os
import re
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
from nn.globals import NN_CFG

@cocotb.test
async def full_cnn_test(dut) -> None:
    """
    Full CNN Integration Test using NNConfig and Hardware Weights.

    Controlled by env vars:
      INJECT_PIXELS : "zeros" | "ones" | "<comma-separated ints>" | "" (dataset)
      SAMPLE_IDX    : dataset index when INJECT_PIXELS is not set (default 10)
    """
    # 1. Load Architecture Config
    config = NN_CFG
    # Apply learned shift from VH file so CNNModel uses the correct ShiftBits
    classifier_shift_env = int(os.environ.get("CLASSIFIER_SHIFT", str(config.classifier_config._shift)))
    config.classifier_config._shift = classifier_shift_env

    # 2. Instantiate Functional Model
    from typing import Any
    weights_dict: dict[str, Any] = {}
    from util.bitwise import unpack_kernel_weights, unpack_weights, unpack_biases

    for i, layer_cfg in enumerate(config.layers):
        cfg = layer_cfg.ConvLayer
        W_int = int(os.environ[f"INJECTED_WEIGHTS_{i}_INT"], 0)
        B_int = int(os.environ[f"INJECTED_BIASES_{i}_INT"], 0)
        if cfg._bias_bits is None or cfg._out_ch is None:
            raise ValueError("Bias bits or out channels must be non-zero")
        weights_dict[f"LAYER_{i}_WEIGHTS"] = unpack_kernel_weights(W_int, cfg._q_schedule._q_min_bits, cfg._out_ch, cfg._in_ch, cfg._kernel_width)
        weights_dict[f"LAYER_{i}_BIASES"] = unpack_biases(B_int, cfg._bias_bits, cfg._out_ch)

    c_idx = len(config.layers)
    c_cfg = config.classifier_config
    CW_int = int(os.environ[f"INJECTED_WEIGHTS_{c_idx}_INT"], 0)
    CB_int = int(os.environ[f"INJECTED_BIASES_{c_idx}_INT"], 0)
    weights_dict[f"LAYER_{c_idx}_WEIGHTS"] = unpack_weights(CW_int, c_cfg._q_schedule._q_min_bits, c_cfg._num_classes, c_cfg._in_ch)
    weights_dict[f"LAYER_{c_idx}_BIASES"] = unpack_biases(CB_int, c_cfg._bias_bits, c_cfg._num_classes)

    functional_model = CNNModel(dut, config, weights_dict)

    # 3. Setup verification components
    if config.in_dims.width is None or config.in_dims.height is None:
        assert False, "CNN input dimensions are not set!"
    n_pixels = config.in_dims.width * config.in_dims.height

    m = ModelRunner(dut, functional_model)
    om = OutputModel(dut, length=1) # Expect 1 classification ID
    im = InputModel(dut, PictureGenerator(functional_model), RateGenerator(dut, 1), n_pixels)

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
    from nn.globals import MNIST_CLASSES
    from nn.inference import get_inference, get_inference_from_pixels
    from nn.sample import get_sample

    class_names = MNIST_CLASSES

    assert len(functional_model.results) > 0, "No hardware output captured — pipeline may have timed out"
    got_id = functional_model.results[0]

    inject_env = os.environ.get("INJECT_PIXELS", "")
    if inject_env == "zeros":
        pixels = [0] * n_pixels
        sw_id  = get_inference_from_pixels(pixels, config)
        label  = None
        desc   = "injected all-zeros"
    elif inject_env == "ones":
        pixels = [1] * n_pixels
        sw_id  = get_inference_from_pixels(pixels, config)
        label  = None
        desc   = "injected all-ones"
    elif inject_env:
        pixels = [int(v) for v in inject_env.split(",")]
        sw_id  = get_inference_from_pixels(pixels, config)
        label  = None
        desc   = f"injected custom ({len(pixels)} px)"
    else:
        sample_idx = int(os.environ.get("SAMPLE_IDX", "10"))
        sw_id  = get_inference(sample_idx)
        _, label = get_sample(sample_idx)
        desc   = f"dataset sample {sample_idx}"

    dut._log.info(f"--- CNN VERIFICATION RESULTS ({desc}) ---")
    dut._log.info(f"Ground Truth:     {label}")
    dut._log.info(f"Hardware Output:  {got_id} ({class_names[got_id]})")
    dut._log.info(f"SW Integer Model: {sw_id} ({class_names[sw_id]})")

    assert got_id == sw_id, (
        f"Hardware/SW mismatch: hardware={got_id} ({class_names[got_id]}), "
        f"sw={sw_id} ({class_names[sw_id]}), input={desc}"
    )
    dut._log.info("SUCCESS: Hardware output matches SW integer prediction!")

def test_full(inject_pixels: str = "") -> None:
    """Pytest entry point.

    Args:
        inject_pixels: "zeros" | "ones" | "<comma-separated ints>" | "" (dataset, default)
    """
    tbpath = Path(__file__).parent
    config = NN_CFG
    simulator = "verilator"
    params = {"BusBits": 8}
    testname = "full_cnn_test"

    # 1. Prepare work directory
    param_str = get_param_string(params)
    work_dir = os.path.join(tbpath, "run", testname, param_str, simulator)
    os.makedirs(work_dir, exist_ok=True)

    # Propagate inject mode to cocotb test via env var
    if inject_pixels:
        os.environ["INJECT_PIXELS"] = inject_pixels
    else:
        os.environ.pop("INJECT_PIXELS", None)
    
    # 2. Load weights from original .vh for injection
    vh_path = (tbpath / ".." / ".." / ".." / "nn" / "data" / "hardware_weights.vh").resolve()
    _, raw_dict = load_weights_from_vh(str(vh_path), config)

    # Parse CLASSIFIER_SHIFT from the vh file (different format: localparam int ...)
    with open(vh_path, 'r') as f:
        vh_content = f.read()
    m = re.search(r"localparam\s+int\s+CLASSIFIER_SHIFT\s*=\s*(\d+)\s*;", vh_content)
    if m is None:
        raise ValueError("CLASSIFIER_SHIFT not found in hardware_weights.vh")
    classifier_shift = int(m.group(1))
    # Stamp learned shift onto config so CNNModel picks it up via classifier_config._shift
    config.classifier_config._shift = classifier_shift
    os.environ["CLASSIFIER_SHIFT"] = str(classifier_shift)

    # 3. Inject parameters and generate headers for each layer
    # Feature Layers
    for i, layer_cfg in enumerate(config.layers):
        cfg = layer_cfg.ConvLayer
        inject_weights_and_biases(
            simulator=simulator, parameters=params, param_str=param_str,
            tbpath=tbpath, test_class="full", Weights=raw_dict[f"LAYER_{i}_WEIGHTS"], Biases=raw_dict[f"LAYER_{i}_BIASES"],
            weight_bits=cfg._q_schedule._q_min_bits, bias_bits=cfg._bias_bits * cfg._out_ch,
            weight_count=cfg._out_ch * cfg._in_ch * (cfg._kernel_width**2),
            layer=i, dsp_count=cfg._dsp_count, custom_work_dir=work_dir,
            oc=cfg._out_ch, ic=cfg._in_ch, kw=cfg._kernel_width)

    # Classifier
    c_idx = len(config.layers)
    c_cfg = config.classifier_config
    inject_weights_and_biases(
        simulator=simulator, parameters=params, param_str=param_str,
        tbpath=tbpath, test_class="full", Weights=raw_dict[f"LAYER_{c_idx}_WEIGHTS"], Biases=raw_dict[f"LAYER_{c_idx}_BIASES"],
        weight_bits=c_cfg._q_schedule._q_min_bits, bias_bits=c_cfg._bias_bits * c_cfg._num_classes,
        weight_count=c_cfg._num_classes * c_cfg._in_ch,
        layer=c_idx, dsp_count=c_cfg._dsp_count, custom_work_dir=work_dir,
        oc=c_cfg._num_classes, ic=c_cfg._in_ch, kw=1)

    # Write layer_N_weights.vh and layer_N_biases.vh with exact RTL-matching bit widths.
    # These may be stale from previous runs with different configs, so always regenerate.
    def _write_vh(path: str, name: str, total_bits: int, raw_val: int) -> None:
        mask = (1 << total_bits) - 1
        packed = raw_val & mask
        hex_w = (total_bits + 3) // 4
        with open(path, "w") as f:
            f.write(f"localparam logic signed [{total_bits-1}:0] {name} = {total_bits}'h{packed:0{hex_w}x};\n")

    for i, layer_cfg in enumerate(config.layers):
        cfg = layer_cfg.ConvLayer
        w_bits = cfg._q_schedule._q_min_bits * cfg._out_ch * cfg._in_ch * (cfg._kernel_width ** 2)
        b_bits = cfg._bias_bits * cfg._out_ch
        _write_vh(os.path.join(work_dir, f"layer_{i}_weights.vh"), f"LAYER_{i}_WEIGHTS", w_bits, raw_dict[f"LAYER_{i}_WEIGHTS"])
        _write_vh(os.path.join(work_dir, f"layer_{i}_biases.vh"),  f"LAYER_{i}_BIASES",  b_bits, raw_dict[f"LAYER_{i}_BIASES"])

    c_w_bits = c_cfg._q_schedule._q_min_bits * c_cfg._num_classes * c_cfg._in_ch
    c_b_bits = c_cfg._bias_bits * c_cfg._num_classes
    _write_vh(os.path.join(work_dir, f"layer_{c_idx}_weights.vh"), f"LAYER_{c_idx}_WEIGHTS", c_w_bits, raw_dict[f"LAYER_{c_idx}_WEIGHTS"])
    _write_vh(os.path.join(work_dir, f"layer_{c_idx}_biases.vh"),  f"LAYER_{c_idx}_BIASES",  c_b_bits, raw_dict[f"LAYER_{c_idx}_BIASES"])

    # Create the master header that includes all layer headers
    # Emit all localparam int lines from the VH (shifts, TruncGuard values, etc.)
    # so that cnn.sv can resolve them as elaboration-time constants.
    int_params = re.findall(r"localparam\s+int\s+\w+\s*=\s*\d+\s*;", vh_content)
    master_header_path = os.path.join(work_dir, "injected_weights_0.vh")
    with open(master_header_path, "w") as f:
        for param_line in int_params:
            f.write(param_line + "\n")
        f.write("\n")
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
        pymodule="test_cnn",
        testname=testname,
        jsonname="cnn.json",
        work_dir=work_dir,
        sim_build=work_dir,
        includes=[work_dir],
    )

if __name__ == "__main__":
    test_full()
