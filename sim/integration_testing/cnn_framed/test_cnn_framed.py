# test_cnn_framed.py
import os
import re
from pathlib import Path
import cocotb
from cocotb.triggers import FallingEdge
import sys

sys.set_int_max_str_digits(0)

from util.utilities  import runner, clock_start_sequence, reset_sequence, get_param_string, inject_weights_and_biases
from util.components import ModelRunner, RateGenerator, InputModel, OutputModel
from functional_models.cnn_model import CNNModel
from functional_models.cnn_framed_model import CnnFramedModel, FramedPictureGenerator
from util.weight_loader import load_weights_from_vh, load_raw_weights_from_csv
from nn.globals import NN_CFG
from nn.verilog import patch_cnn_framed_sv

@cocotb.test
async def full_cnn_test(dut) -> None:
    """
    End-to-end integration test for cnn_framed (deframer + CNN + class_framer).

    Drives the byte-level framed interface just like the Python demo:
      [0xA5, 0x5A] header  +  9600 packed image bytes  →  DUT  →  [0x99, class, 0xA5, 0x5A]

    The class_framer always emits one 0x99 wakeup byte before the first class
    result. The ModelRunner sees an empty queue at that point and skips it
    automatically. The subsequent [class_id, 0xA5, 0x5A] bytes are verified
    against the SW integer functional model.

    Controlled by env vars:
      INJECT_PIXELS : "zeros" | "ones" | "<comma-separated ints>" | "" (dataset)
      SAMPLE_IDX    : dataset index when INJECT_PIXELS is not set (default 10)
    """
    from util.bitwise import unpack_kernel_weights, unpack_weights, unpack_biases
    from nn.sample import get_sample
    from nn.globals import CLASSES as class_names_list
    from nn.inference import get_inference, get_inference_from_pixels

    # 1. Architecture config + learned shift
    config = NN_CFG
    last_layer_shift_key = f"LAYER_{len(config.layers) - 1}_SHIFT"
    config.classifier_config._shift = int(
        os.environ.get(last_layer_shift_key, str(config.classifier_config._shift))
    )

    # 2. Reconstruct weights dict from injected env vars
    weights_dict = {}
    for i, layer_cfg in enumerate(config.layers):
        cfg = layer_cfg.ConvLayer
        W_int = int(os.environ[f"INJECTED_WEIGHTS_{i}_INT"], 0)
        B_int = int(os.environ[f"INJECTED_BIASES_{i}_INT"], 0)
        weights_dict[f"LAYER_{i}_WEIGHTS"] = unpack_kernel_weights(
            W_int, cfg._q_schedule._q_min_bits, cfg._out_ch, cfg._in_ch, cfg._kernel_width
        )
        weights_dict[f"LAYER_{i}_BIASES"] = unpack_biases(B_int, cfg._bias_bits, cfg._out_ch)

    c_idx = len(config.layers)
    c_cfg = config.classifier_config
    CW_int = int(os.environ[f"INJECTED_WEIGHTS_{c_idx}_INT"], 0)
    CB_int = int(os.environ[f"INJECTED_BIASES_{c_idx}_INT"], 0)
    weights_dict[f"LAYER_{c_idx}_WEIGHTS"] = unpack_weights(
        CW_int, c_cfg._q_schedule._q_min_bits, c_cfg._num_classes, c_cfg._in_ch
    )
    weights_dict[f"LAYER_{c_idx}_BIASES"] = unpack_biases(CB_int, c_cfg._bias_bits, c_cfg._num_classes)

    # 3. Standalone CNN functional model (not bound to dut ports)
    cnn = CNNModel(None, config, weights_dict)

    # 4. Load pixels
    assert config.in_dims.height is not None and config.in_dims.width is not None, "Input dimensions must be specified in config"
    width = int(config.in_dims.width)
    height = int(config.in_dims.height)
    n_pixels = width * height
    inject_env = os.environ.get("INJECT_PIXELS", "")
    if inject_env == "zeros":
        pixels, label, desc = [0] * n_pixels, None, "injected all-zeros"
    elif inject_env == "ones":
        pixels, label, desc = [1] * n_pixels, None, "injected all-ones"
    elif inject_env:
        pixels, label, desc = (
            [int(v) for v in inject_env.split(",")],
            None,
            f"injected custom ({len(inject_env.split(','))} px)",
        )
    else:
        sample_idx = int(os.environ.get("SAMPLE_IDX", "10"))
        pixels, label = get_sample(sample_idx)
        desc = f"dataset sample {sample_idx}"

    # 5. Framed functional model wraps dut + cnn
    packet_len_bytes = (width * height) // 8
    framed_model = CnnFramedModel(dut, cnn, packet_len_bytes=packet_len_bytes)

    # 6. Framed byte-level input generator
    assert pixels is not None, "Pixels must be loaded before creating FramedPictureGenerator"
    framed_gen = FramedPictureGenerator(pixels, bus_bits=8)

    # 7. Wire up components
    #    ModelRunner handles input handshakes (valid_i/ready_o) → consume()
    #    and output handshakes (valid_o/ready_i) → produce().
    #    The wakeup 0x99 fires with empty queue → silently skipped by ModelRunner.
    #    OutputModel drives ready_i=1 and counts all 4 output bytes
    #    (0x99 + class_id + 0xA5 + 0x5A).
    m  = ModelRunner(dut, framed_model)
    im = InputModel(dut, framed_gen, RateGenerator(dut, 1), framed_gen.total_bytes())
    om = OutputModel(dut, length=4)  # wakeup + class + tail0 + tail1

    # 8. Execute simulation
    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)
    await FallingEdge(dut.clk_i)

    m.start()
    om.start()
    im.start()

    timeout_ns = n_pixels * 100
    await om.wait(timeout_ns)

    # 9. Report and assert
    class_names = sorted(class_names_list)
    assert len(framed_model.results) > 0, (
        "No class byte captured — CnnFramedModel.produce() was never called with class position. "
        "Pipeline may have timed out or class_framer wakeup byte consumed all produce() calls."
    )
    got_id = framed_model.results[0]

    if inject_env == "zeros":
        sw_id = get_inference_from_pixels([0] * n_pixels, config)
    elif inject_env == "ones":
        sw_id = get_inference_from_pixels([1] * n_pixels, config)
    elif inject_env:
        sw_id = get_inference_from_pixels(pixels, config)
    else:
        sw_id = get_inference(int(os.environ.get("SAMPLE_IDX", "10")))

    dut._log.info(f"--- CNN FRAMED VERIFICATION ({desc}) ---")
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
    assert config.in_dims.width is not None and config.in_dims.height is not None
    simulator = "verilator"
    params = {"BusBits": 8, "WidthIn": config.in_dims.width, "HeightIn": config.in_dims.height}
    testname = "full_cnn_test"

    param_str = get_param_string(params)
    work_dir = os.path.join(tbpath, "run", testname, param_str, simulator)
    os.makedirs(work_dir, exist_ok=True)

    if inject_pixels:
        os.environ["INJECT_PIXELS"] = inject_pixels
    else:
        os.environ.pop("INJECT_PIXELS", None)

    # Load weights from hardware_weights.vh
    vh_path = (tbpath / ".." / ".." / ".." / "nn" / "data" / "hardware_weights.vh").resolve()
    _, raw_dict = load_weights_from_vh(str(vh_path), config)
    csv_path = (tbpath / ".." / ".." / ".." / "nn" / "data" / "hardware_weights.csv").resolve()
    csv_weights = load_raw_weights_from_csv(str(csv_path), config)
    for k, v in csv_weights.items():
        raw_dict.setdefault(k, v)

    with open(vh_path) as f:
        vh_content = f.read()
    last_n = len(config.layers) - 1
    last_layer_shift_key = f"LAYER_{last_n}_SHIFT"
    m = re.search(rf"localparam\s+int\s+{last_layer_shift_key}\s*=\s*(\d+)\s*;", vh_content)
    if m is None:
        raise ValueError(f"{last_layer_shift_key} not found in hardware_weights.vh")
    last_layer_shift = int(m.group(1))
    config.classifier_config._shift = last_layer_shift
    os.environ[last_layer_shift_key] = str(last_layer_shift)

    # Sync tb_cnn_framed.sv FileName parameters to current model's ROM layout
    patch_cnn_framed_sv(config, tbpath / "tb_cnn_framed.sv")

    # Inject weights for all conv/pool layers
    for i, layer_cfg in enumerate(config.layers):
        cfg = layer_cfg.ConvLayer
        inject_weights_and_biases(
            simulator=simulator, parameters=params, param_str=param_str,
            tbpath=tbpath, test_class="framed",
            Weights=raw_dict[f"LAYER_{i}_WEIGHTS"], Biases=raw_dict[f"LAYER_{i}_BIASES"],
            weight_bits=cfg._q_schedule._q_min_bits, bias_bits=cfg._bias_bits,
            weight_count=cfg._out_ch * cfg._in_ch * (cfg._kernel_width ** 2),
            layer=i, dsp_count=cfg._dsp_count, custom_work_dir=work_dir,
            oc=cfg._out_ch, ic=cfg._in_ch, kw=cfg._kernel_width,
        )

    # Inject classifier weights
    c_idx = len(config.layers)
    c_cfg = config.classifier_config
    inject_weights_and_biases(
        simulator=simulator, parameters=params, param_str=param_str,
        tbpath=tbpath, test_class="framed",
        Weights=raw_dict[f"LAYER_{c_idx}_WEIGHTS"], Biases=raw_dict[f"LAYER_{c_idx}_BIASES"],
        weight_bits=c_cfg._q_schedule._q_min_bits, bias_bits=c_cfg._bias_bits,
        weight_count=c_cfg._num_classes * c_cfg._in_ch,
        layer=c_idx, dsp_count=c_cfg._dsp_count, custom_work_dir=work_dir,
        oc=c_cfg._num_classes, ic=c_cfg._in_ch, kw=1,
    )

    # Write per-layer .vh headers for the RTL include
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

    # Master header: all localparam int lines from vh (shifts, TruncGuard, etc.) + layer includes
    int_params = re.findall(r"localparam\s+int\s+\w+\s*=\s*\d+\s*;", vh_content)
    master_header_path = os.path.join(work_dir, "injected_weights_0.vh")
    with open(master_header_path, "w") as f:
        for param_line in int_params:
            f.write(param_line + "\n")
        f.write("\n")
        for i in range(len(config.layers) + 1):
            f.write(f'`include "layer_{i}_weights.vh"\n')
            f.write(f'`include "layer_{i}_biases.vh"\n')

    os.environ["CHECK_INFERENCE"] = os.environ.get("CHECK_INFERENCE", "1")
    os.environ["SAMPLE_IDX"]      = os.environ.get("SAMPLE_IDX", "10")

    runner(
        simulator=simulator,
        timescale="1ps/1ps",
        tbpath=tbpath,
        params=params,
        pymodule="test_cnn_framed",
        testname=testname,
        jsonname="cnn_framed.json",
        work_dir=work_dir,
        sim_build=work_dir,
        includes=[work_dir],
        toplevel_override="cnn_framed",
    )


if __name__ == "__main__":
    test_full()
