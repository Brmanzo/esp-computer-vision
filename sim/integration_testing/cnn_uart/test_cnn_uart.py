# test_cnn_uart.py
# Full end-to-end test for uart_cnn.sv: sends a framed image over a simulated
# UART serial link and verifies the class response, exactly as python_demo.py does
# on real hardware.
import os
import re
from pathlib import Path
import cocotb
from cocotb.triggers import with_timeout
import sys

sys.set_int_max_str_digits(0)

from cocotbext.uart import UartSource, UartSink

from util.utilities  import runner, clock_start_sequence, reset_sequence, get_param_string, inject_weights_and_biases
from util.weight_loader import load_weights_from_vh
from functional_models.cnn import CNNModel
from nn.globals import NN_CFG
from nn.protocol import build_frame, parse_response

# uart_cnn.sv has prescale=13 hardcoded.
# At the 1 GHz simulation clock (1 ns/cycle):
#   bit_period = 8 * prescale * 1 ns = 104 ns
#   baud = 1e9 / 104 ≈ 9_615_384
_CLK_PERIOD_NS = 1
_PRESCALE      = 13
BAUD           = int(1e9 / (_CLK_PERIOD_NS * 8 * _PRESCALE))  # ≈ 9_615_384

# Timeout for the entire 4-byte response:
#   TX: 9602 bytes × 1040 ns + CNN pipeline + 4 RX bytes × 1040 ns ≈ 10.1 ms
RESPONSE_TIMEOUT_NS = 15_000_000  # 15 ms — generous margin


@cocotb.test
async def full_cnn_test(dut) -> None:
    """
    End-to-end UART integration test for uart_cnn.sv.

    Transmits [0xA5 0x5A | 9600 packed image bytes] over rx_serial_i using
    cocotbext-uart, then receives the 4-byte response on tx_serial_o:
      0x99  (wakeup, sent immediately after reset)
      class (CNN classification result)
      0xA5  (tail byte 0)
      0x5A  (tail byte 1)

    Verifies that the hardware class byte matches the SW integer model.

    Env vars:
      INJECT_PIXELS : "zeros" | "ones" | "" (dataset sample, default)
      SAMPLE_IDX    : dataset index (default 10)
    """
    from util.bitwise import unpack_kernel_weights, unpack_weights, unpack_biases
    from nn.sample import get_sample
    from nn.tasks.hand_gesture.preprocess import get_class_names
    from nn.inference import get_inference, get_inference_from_pixels

    # 1. Architecture config + learned shift
    config = NN_CFG
    config.classifier_config._shift = int(
        os.environ.get("CLASSIFIER_SHIFT", str(config.classifier_config._shift))
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

    # 3. Standalone CNN functional model for SW reference prediction
    cnn = CNNModel(None, config, weights_dict)

    # 4. Load pixels
    assert config.in_dims.height is not None and config.in_dims.width is not None
    width, height = int(config.in_dims.width), int(config.in_dims.height)
    n_pixels = width * height
    inject_env = os.environ.get("INJECT_PIXELS", "")
    if inject_env == "zeros":
        pixels, label, desc = [0] * n_pixels, None, "injected all-zeros"
    elif inject_env == "ones":
        pixels, label, desc = [1] * n_pixels, None, "injected all-ones"
    else:
        sample_idx = int(os.environ.get("SAMPLE_IDX", "10"))
        pixels, label = get_sample(sample_idx)
        desc = f"dataset sample {sample_idx}"

    # 5. Compute SW prediction (before simulation)
    assert pixels is not None, "Pixels must be loaded before computing SW prediction"
    if inject_env in ("zeros", "ones"):
        sw_id = get_inference_from_pixels(pixels, config)
    else:
        sw_id = get_inference(int(os.environ.get("SAMPLE_IDX", "10")))

    # 6. Build the framed byte sequence (header + packed pixels)
    assert pixels is not None
    framed_bytes = build_frame(pixels)

    # 7. Attach cocotbext-uart source and sink
    #    UartSource drives rx_serial_i; UartSink monitors tx_serial_o.
    uart_source = UartSource(dut.rx_serial_i, baud=BAUD, bits=8)
    uart_sink   = UartSink  (dut.tx_serial_o, baud=BAUD, bits=8)

    # 8. Clock + reset
    await clock_start_sequence(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i, 10)

    # 9. Send framed image and receive response concurrently.
    #    The class_framer sends 0x99 immediately after reset (before any image
    #    data arrives), so the sink task must be running from the start.
    async def recv_4_bytes():
        resp = bytearray()
        for _ in range(4):
            await with_timeout(uart_sink.wait(), RESPONSE_TIMEOUT_NS, "ns")
            resp.extend(uart_sink.read_nowait(1))
        return resp

    rx_task = cocotb.start_soon(recv_4_bytes())
    await uart_source.write(framed_bytes)
    await uart_source.wait()          # ensure all bits clocked out

    resp = await rx_task

    # 10. Parse response using the same logic as python_demo.py
    dut._log.info(f"Raw response: {' '.join(f'0x{b:02X}' for b in resp)}")
    got_id = parse_response(bytes(resp))

    # 11. Report and assert
    class_names = get_class_names()
    dut._log.info(f"--- CNN UART VERIFICATION ({desc}) ---")
    dut._log.info(f"Ground Truth:     {label}")
    dut._log.info(f"Hardware Output:  {got_id} ({class_names[got_id]})")
    dut._log.info(f"SW Integer Model: {sw_id} ({class_names[sw_id]})")

    assert got_id == sw_id, (
        f"Hardware/SW mismatch: hardware={got_id} ({class_names[got_id]}), "
        f"sw={sw_id} ({class_names[sw_id]}), input={desc}"
    )
    dut._log.info("SUCCESS: Hardware output matches SW integer prediction!")


def test_full(inject_pixels: str = "") -> None:
    """Pytest entry point."""
    tbpath = Path(__file__).parent
    config = NN_CFG
    simulator = "verilator"
    params = {"BusBits": 8}
    testname = "full_cnn_test"

    param_str = get_param_string(params)
    work_dir = os.path.join(tbpath, "run", testname, param_str, simulator)
    os.makedirs(work_dir, exist_ok=True)

    if inject_pixels:
        os.environ["INJECT_PIXELS"] = inject_pixels
    else:
        os.environ.pop("INJECT_PIXELS", None)

    # Load weights from hardware_weights.vh
    vh_path = (tbpath / ".." / ".." / ".." / "model" / "data" / "hardware_weights.vh").resolve()
    _, raw_dict = load_weights_from_vh(str(vh_path), config)

    with open(vh_path) as f:
        vh_content = f.read()
    m = re.search(r"localparam\s+int\s+CLASSIFIER_SHIFT\s*=\s*(\d+)\s*;", vh_content)
    if m is None:
        raise ValueError("CLASSIFIER_SHIFT not found in hardware_weights.vh")
    classifier_shift = int(m.group(1))
    config.classifier_config._shift = classifier_shift
    os.environ["CLASSIFIER_SHIFT"] = str(classifier_shift)

    # Inject weights for all conv/pool layers
    for i, layer_cfg in enumerate(config.layers):
        cfg = layer_cfg.ConvLayer
        inject_weights_and_biases(
            simulator=simulator, parameters=params, param_str=param_str,
            tbpath=tbpath, test_class="uart",
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
        tbpath=tbpath, test_class="uart",
        Weights=raw_dict[f"LAYER_{c_idx}_WEIGHTS"], Biases=raw_dict[f"LAYER_{c_idx}_BIASES"],
        weight_bits=c_cfg._q_schedule._q_min_bits, bias_bits=c_cfg._bias_bits,
        weight_count=c_cfg._num_classes * c_cfg._in_ch,
        layer=c_idx, dsp_count=c_cfg._dsp_count, custom_work_dir=work_dir,
        oc=c_cfg._num_classes, ic=c_cfg._in_ch, kw=1,
    )

    # Write per-layer .vh headers for the RTL include (cnn.sv's COCOTB_SIM guard)
    def _write_vh(path: str, name: str, total_bits: int, raw_val: int) -> None:
        mask   = (1 << total_bits) - 1
        packed = raw_val & mask
        hex_w  = (total_bits + 3) // 4
        with open(path, "w") as f:
            f.write(f"localparam logic signed [{total_bits-1}:0] {name} = {total_bits}'h{packed:0{hex_w}x};\n")

    for i, layer_cfg in enumerate(config.layers):
        cfg    = layer_cfg.ConvLayer
        w_bits = cfg._q_schedule._q_min_bits * cfg._out_ch * cfg._in_ch * (cfg._kernel_width ** 2)
        b_bits = cfg._bias_bits * cfg._out_ch
        _write_vh(os.path.join(work_dir, f"layer_{i}_weights.vh"), f"LAYER_{i}_WEIGHTS", w_bits, raw_dict[f"LAYER_{i}_WEIGHTS"])
        _write_vh(os.path.join(work_dir, f"layer_{i}_biases.vh"),  f"LAYER_{i}_BIASES",  b_bits, raw_dict[f"LAYER_{i}_BIASES"])

    c_w_bits = c_cfg._q_schedule._q_min_bits * c_cfg._num_classes * c_cfg._in_ch
    c_b_bits = c_cfg._bias_bits * c_cfg._num_classes
    _write_vh(os.path.join(work_dir, f"layer_{c_idx}_weights.vh"), f"LAYER_{c_idx}_WEIGHTS", c_w_bits, raw_dict[f"LAYER_{c_idx}_WEIGHTS"])
    _write_vh(os.path.join(work_dir, f"layer_{c_idx}_biases.vh"),  f"LAYER_{c_idx}_BIASES",  c_b_bits, raw_dict[f"LAYER_{c_idx}_BIASES"])

    # Master header: CLASSIFIER_SHIFT + all layer .vh includes
    with open(os.path.join(work_dir, "injected_weights_0.vh"), "w") as f:
        f.write(f"localparam int CLASSIFIER_SHIFT = {classifier_shift};\n")
        for i in range(len(config.layers) + 1):
            f.write(f'`include "layer_{i}_weights.vh"\n')
            f.write(f'`include "layer_{i}_biases.vh"\n')

    os.environ["SAMPLE_IDX"] = os.environ.get("SAMPLE_IDX", "10")

    runner(
        simulator=simulator,
        timescale="1ps/1ps",
        tbpath=tbpath,
        params=params,
        pymodule="test_cnn_uart",
        testname=testname,
        jsonname="cnn_uart.json",
        work_dir=work_dir,
        sim_build=work_dir,
        includes=[work_dir],
    )


if __name__ == "__main__":
    test_full()
