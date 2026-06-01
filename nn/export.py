#!/usr/bin/env python3
# nn.export.py
# Usage: cnn.py export [network_in.pth] [hardware_weights.csv] [hardware_weights.vh] [hex_output_dir] [--random]

import argparse
import ast
import os
import math
import pandas as pd
import sys
import torch
from   pathlib  import Path
from   datetime import datetime
from   typing   import Any, Optional

from nn.arch     import cnn, QuantConv2d
from nn.config   import NNConfig, ConvConfig, ClassifierConfig
from nn.globals  import DATAPATH, ROMPATH, NN_CFG, NET_PATH
from nn.quantize import LearnedShiftQuantizer, QuantizeActivation, \
                        generate_random_quantized_weights

def compute_min_trunc_guard(
    weights_flat: list, biases: list,
    in_bits: int, out_bits: int,
    in_channels: int, out_channels: int, kernel_width: int,
    learned_shift: int,
) -> int:
    '''Return the minimum TruncGuard such that AccBits covers the worst-case
    accumulation for these actual quantized weights (signed, from the CSV).

    For binary inputs (in_bits==1) the hardware maps 0→-1 and 1→+1, so each
    term contributes ±weight; worst-case magnitude per OC is Σ|w|.
    For unsigned inputs (in_bits>2) worst-case positive acc is max_input x Σmax(w,0)
    and worst-case negative is max_input x Σmin(w,0).

    The minimum AccBits to hold the signed worst-case value without wrap:
        acc_bits_needed = ceil(log2(worst_mag + 1)) + 1
    Then:
        min_trunc_guard = max(0, acc_bits_needed - learned_shift - out_bits)
    '''
    import numpy as np
    w = np.array(weights_flat, dtype=np.int64).reshape(out_channels, in_channels, kernel_width, kernel_width)
    b = np.array(biases, dtype=np.int64)

    unsigned = in_bits > 2
    max_input = (1 << in_bits) - 1 if unsigned else 1  # binary: effective magnitude is 1

    worst_mag = 0
    for oc in range(out_channels):
        w_oc = w[oc]
        if not unsigned:
            # Binary (in_bits=1) and ternary (in_bits=2): each input can be ±1, so
            # worst-case positive is achieved by aligning input sign with weight sign → Σ|w|.
            max_pos = int(np.sum(np.abs(w_oc))) + int(b[oc])
            min_neg = -int(np.sum(np.abs(w_oc))) + int(b[oc])
        else:
            # Unsigned input [0, max_input]: maximise positive by using max_input for
            # positive weights and 0 for negative weights (and vice-versa for min).
            max_pos = max_input * int(np.sum(np.maximum(w_oc, 0))) + int(b[oc])
            min_neg = max_input * int(np.sum(np.minimum(w_oc, 0))) + int(b[oc])
        worst_mag = max(worst_mag, abs(max_pos), abs(min_neg))

    if worst_mag <= 1:
        acc_bits_needed = 2
    else:
        acc_bits_needed = math.ceil(math.log2(worst_mag + 1)) + 1  # +1 for sign bit

    return max(0, acc_bits_needed - learned_shift - out_bits)


def get_hardware_weights(fused_layer: QuantConv2d, previous_act_scale: float = 1.0):
    '''Extracts the folded integer weights and biases directly from the QAT layer.
    
    Applies multi-bit activation scale compensation:
    y = W * (x_hw * act_scale) + b  →  (W * act_scale) * x_hw + b
    '''

    with torch.no_grad():
        # 1. Recreate the exact BN fold that happened during training
        bn_scale = fused_layer.bn_weight / torch.sqrt(fused_layer.running_var + fused_layer.eps)
        folded_w = fused_layer.weight * bn_scale.view(-1, 1, 1, 1)
        folded_b = fused_layer.bn_bias - (fused_layer.running_mean * bn_scale)

        # Factor in the scale of the activations coming into this layer!
        folded_w = folded_w * previous_act_scale

        # 2. Retrieve the target bits directly from the layer
        target_bits = int(fused_layer._weight_bits)
        bias_bits = getattr(fused_layer, '_bias_bits', 8)

        # 3. Quantize to hardware integers
        qmin = -(2 ** (target_bits - 1))
        qmax = (2 ** (target_bits - 1)) - 1

        max_abs = folded_w.abs().max()
        raw_scale = max_abs / qmax if max_abs > 0 else 1.0
        w_scale = 2.0 ** torch.round(torch.log2(torch.as_tensor(raw_scale, device=folded_w.device)))

        q_weights = torch.round(folded_w / w_scale)
        q_weights = torch.clip(q_weights, qmin, qmax)

        hw_bias = torch.round(folded_b / w_scale)

        # Safety: clamp biases to hardware bit-width
        b_max_val = 2**(bias_bits - 1) - 1
        b_min_val = -2**(bias_bits - 1)

        if (hw_bias > b_max_val).any() or (hw_bias < b_min_val).any():
            print(f"  [WARNING] Bias saturating! Max={hw_bias.abs().max().item():.0f}, Limit={b_max_val}")
            hw_bias = torch.clamp(hw_bias, b_min_val, b_max_val)

        return q_weights, hw_bias, float(w_scale.item() if torch.is_tensor(w_scale) else w_scale)

def export_nn_to_csv(nn_path: Path, config: NNConfig, output_csv: Path, random_weights: bool = False):
    '''Loads the trained network, extracts the folded and quantized weights for
    each layer, and saves them to a CSV file for hardware implementation.
    If random_weights is True, it instead generates random hardware weights
    constrained by the network config's quantization schedule.'''
    device = "cpu"

    network = cnn(config=config).to(device)
    if not random_weights:
        state = torch.load(nn_path, map_location=device, weights_only=True)
        network.load_state_dict(state, strict=True)

    # Activate quantization at q_min_bits for every conv layer so that
    # get_hardware_weights sees the same quantized weights and BN statistics
    # that the model used during the final [Q] plateau of training.
    quant_idx = 0
    from nn.quantize import QuantConv2d as _QC
    for module in network.modules():
        if isinstance(module, _QC):
            module._quantize    = True
            module._weight_bits = config.q_schedule[quant_idx]._q_min_bits
            quant_idx += 1

    network.eval()

    all_hardware_data: list[dict[str, Any]] = []
    current_act_scale = 1.0 # Input pixels are binarized {0, 1} or normalized, start at 1.0
    last_w_scale      = 1.0 # Tracks w_scale of the most recent QuantConv2d, needed for LearnedShiftQuantizer

    print("Exporting Fused QAT layers to hardware CSV with scale propagation...")

    # We must iterate through all modules to track activation scales correctly
    # Features followed by Classifier
    all_modules = list(network.features.children()) + list(network.classifier.children())

    for module in all_modules:
        if isinstance(module, QuantConv2d):
            layer_idx = len(all_hardware_data)
            name = f"Layer_{layer_idx}"
            bias_bits = getattr(module, '_bias_bits', 8)
            
            if random_weights:
                weight_bits = config.q_schedule[layer_idx].q_min_bits
                print(f"  Generating random weights for {name} ({module.__class__.__name__}) with {weight_bits}-bit weights...")
                
                w_shape = module.weight.shape
                w_int_tensor = generate_random_quantized_weights(w_shape, weight_bits)
                b_hw = generate_random_quantized_weights((w_shape[0],), bias_bits)
                w_scale = 1.0
            else:
                weight_bits = module._weight_bits
                print(f"  Processing {name} ({module.__class__.__name__}) with incoming scale {current_act_scale:.6f}...")
                
                # Extract compensated weights
                w_int_tensor, b_hw, w_scale = get_hardware_weights(module, current_act_scale)
                last_w_scale = w_scale

            # Convert to numpy
            w_int = w_int_tensor.to(torch.int32).cpu().numpy()
            b_int = b_hw.to(torch.int32).cpu().numpy()

            all_hardware_data.append({
                "layer_name": name,
                "weight_scale": w_scale,
                "weights_shape": str(tuple(w_int.shape)),
                "weight_bits" : weight_bits,
                "weights_flat": w_int.flatten().tolist(),
                "bias_bits" : bias_bits,
                "bias_flat": b_int.flatten().tolist(),
            })
        
        elif isinstance(module, QuantizeActivation):
            if module.bits == 1:
                # 1-bit activation in PyTorch is {-1, 1}.
                # In hardware it is {0, 1}.
                # The effective scale remains 1.0 because the weights/inputs
                # are handled by XNOR logic which inherently maps {0, 1} back to {-1, 1}.
                current_act_scale = 1.0
                if all_hardware_data:
                    all_hardware_data[-1]["layer_shift"] = 0
            else:
                # Ternary/signed multi-bit: hardware emits sign(acc) ∈ {-1,0,+1} whose
                # integer magnitude is 1, but the float training value is clip/qmax per step.
                # Do NOT stamp layer_shift here — hardware uses sign(acc), not a right-shift,
                # so layer_shift would be read by _layer_acc_bits as shift_override and would
                # collapse the accumulator width to ~5 bits (0+ob+trunc_guard), wrapping
                # a ~10-bit accumulator and producing garbage.
                qmax = (2 ** (module.bits - 1)) - 1
                if module.clip_val is not None:
                    clip = float(module.clip_val.abs().item())
                    current_act_scale = clip / qmax
                    print(f"  Detected Activation Scaling: {current_act_scale:.6f} (clip={clip:.3f}, qmax={qmax})")
                    if all_hardware_data:
                        all_hardware_data[-1]["layer_shift"] = 0
                else:
                    current_act_scale = 2.0 ** module.shift
                    print(f"  Detected Activation Scaling: {current_act_scale:.6f} (Fixed Shift {module.shift})")
                    if all_hardware_data:
                        all_hardware_data[-1]["layer_shift"] = module.shift

        elif isinstance(module, LearnedShiftQuantizer):
            # Derive shift from the ratio of the float-domain clip boundary to the
            # integer accumulator scale.  last_w_scale captures the actual trained weight
            # magnitude, which is far below the analytical worst-case; using the analytical
            # hardware_shift (which ignores w_scale) overestimates the shift and crushes
            # activations to zero.  Use _pow2_clip() for the clip value so it matches what
            # the training forward pass used.
            qmax_out        = (2 ** module._out_bits) - 1
            clip_pow2_float = float(module._pow2_clip().item())
            raw_shift       = math.log2(clip_pow2_float / (qmax_out * last_w_scale))
            # floor() keeps more dynamic range; round() over-compresses when accumulations
            # fall below the clip boundary.
            shift_val       = max(0, math.floor(raw_shift))
            # act_scale = clip_pow2/qmax is what the training forward pass (LearnedShiftQuantizer)
            # maps each integer step to in float space; use it for the next layer's weight compensation.
            current_act_scale = module.act_scale

            if all_hardware_data:
                conv_idx = len(all_hardware_data) - 1
                is_last_feature = (conv_idx == len(config.layers) - 1)
                all_hardware_data[-1]["layer_shift"] = shift_val
                if is_last_feature:
                    config.classifier_config._shift = shift_val
                    print(f"  Learned Last-Layer Shift: {shift_val} (formula={raw_shift:.2f})  act_scale={current_act_scale:.6f}  clip_pow2={clip_pow2_float:.4f}")
                else:
                    print(f"  Layer {conv_idx} Shift: {shift_val} (formula={raw_shift:.2f})  act_scale={current_act_scale:.6f}  clip_pow2={clip_pow2_float:.4f}")

    # If no LearnedShiftQuantizer was encountered (e.g. random_weights path),
    # fall back to the analytically-derived shift on the last feature conv row (random_weights path)
    feature_row_idx = len(all_hardware_data) - 2  # last feature row, not classifier
    if 0 <= feature_row_idx < len(all_hardware_data):
        if "layer_shift" not in all_hardware_data[feature_row_idx]:
            all_hardware_data[feature_row_idx]["layer_shift"] = config.classifier_config._shift

    df = pd.DataFrame(all_hardware_data)
    df.to_csv(output_csv, index=False)
    print(f"\nHardware export complete! Saved to {output_csv}")

def pack_hex_weights(hex_path: Path, weights_flat: list, weight_bits: int, dsp_count: int, cfg: Any, layer_idx: int):
    '''Slices a flat list of weights and writes them to a hex file for $readmemh,
    packing multiple weights per line if multiple DSPs are used in parallel.
    This logic MUST match the RTL address generation in filter_seq.sv and utilities.py.
    '''
    # 1. Extract dimensions
    if isinstance(cfg, ConvConfig):
        oc = cfg._out_ch
        ic = cfg._in_ch
        ka = cfg._kernel_width**2
    elif isinstance(cfg, ClassifierConfig):
        oc = cfg._num_classes
        ic = cfg._in_ch
        ka = 1
        
    # 2. Calculate hardware-accurate partitioning
    effective_dsps  = min(dsp_count, oc)
    neurons_per_dsp = (oc // effective_dsps) if effective_dsps > 0 else 0
    # Ensure integer arithmetic for static type checkers
    total_terms     = int(ic) * int(ka)
    
    rom_width_bits = weight_bits * effective_dsps
    
    # 3. Setup file path and formatting
    filename = f"layer_{layer_idx}_weights.hex"
    hex_width = (rom_width_bits + 3) // 4
    mask = (1 << weight_bits) - 1
    
    # 4. Pack and write: for each local neuron in workload, for each term, pack all DSPs
    os.makedirs(hex_path, exist_ok=True)
    total_words = neurons_per_dsp * total_terms
    # Pad to next multiple of 256 for SB_RAM40_4K tile boundary
    pad_to = ((total_words + 255) // 256) * 256

    # SB_RAM40_4K max width is 16 bits; split wider ROMs into _lo / _hi hex files.
    # lo tile: 16 bits. hi tile: rom_width_bits - 16 bits (NOT a second 16-bit tile).
    # This ensures {hi, lo} reconstructs the full rom_width_bits word in RTL.
    TILE_BITS = 16
    if rom_width_bits > TILE_BITS:
        hi_bits    = rom_width_bits - TILE_BITS
        hi_mask    = (1 << hi_bits) - 1
        hi_hex_w   = (hi_bits + 3) // 4
        lo_hex_w   = TILE_BITS // 4
        lo_mask    = (1 << TILE_BITS) - 1
        filename_lo = f"layer_{layer_idx}_weights_lo.hex"
        filename_hi = f"layer_{layer_idx}_weights_hi.hex"
        # Remove stale single-file ROM if it exists (old format before split was needed)
        stale_single = hex_path / filename
        if stale_single.exists():
            stale_single.unlink()
            print(f"  Removed stale single-file ROM: {filename}")
        with open(hex_path / filename_lo, "w") as f_lo, \
             open(hex_path / filename_hi, "w") as f_hi:
            for local_neuron in range(neurons_per_dsp):
                for term in range(total_terms):
                    packed = 0
                    for dsp_idx in range(effective_dsps):
                        global_oc = dsp_idx * neurons_per_dsp + local_neuron
                        if global_oc < oc:
                            weight = int(weights_flat[global_oc * total_terms + term]) & mask
                            packed |= weight << (dsp_idx * weight_bits)
                    f_lo.write(f"{packed & lo_mask:0{lo_hex_w}x}\n")
                    f_hi.write(f"{(packed >> TILE_BITS) & hi_mask:0{hi_hex_w}x}\n")
            zero_lo = f"{'0' * lo_hex_w}\n"
            zero_hi = f"{'0' * hi_hex_w}\n"
            for _ in range(pad_to - total_words):
                f_lo.write(zero_lo)
                f_hi.write(zero_hi)
        print(f"  Generated ROM hex: {filename_lo} + {filename_hi} ({total_words} words + {pad_to - total_words} pad → {pad_to})")
        return

    # Remove stale split-file ROMs if they exist (old format after switching back to single)
    for stale in [hex_path / f"layer_{layer_idx}_weights_lo.hex",
                  hex_path / f"layer_{layer_idx}_weights_hi.hex"]:
        if stale.exists():
            stale.unlink()
            print(f"  Removed stale split-file ROM: {stale.name}")

    with open(hex_path / filename, "w") as f:
        for local_neuron in range(neurons_per_dsp):
            for term in range(total_terms):
                packed_line = 0
                for dsp_idx in range(effective_dsps):
                    global_oc = dsp_idx * neurons_per_dsp + local_neuron
                    if global_oc < oc:
                        idx = global_oc * total_terms + term
                        # Weights in list are signed integers, mask them for hex
                        weight = int(weights_flat[idx]) & mask
                        packed_line |= (weight << (dsp_idx * weight_bits))
                f.write(f"{packed_line:0{hex_width}x}\n")
        for _ in range(pad_to - total_words):
            f.write(f"{0:0{hex_width}x}\n")

    print(f"  Generated ROM hex: {filename} ({total_words} words + {pad_to - total_words} pad → {pad_to})")

def export_csv_to_hex(csv_path: Path, sv_path: Path, hex_path: Path, config: NNConfig):
    '''Loads the CSV generated by export_nn_to_csv and generates the SystemVerilog
    Header File and Hex files for hardware implementation.
    '''
    date = datetime.now().strftime("%B %d, %Y %I:%M %p")
    sv_lines = [
        "// Auto-generated hardware parameters",
        f"// Date: {date}\n"
    ]
    
    work_dir = sv_path.parent
    
    # Force columns to be strings to avoid Pandas guessing types (like Timedelta)
    df = pd.read_csv(csv_path, dtype={"layer_name": str, "weight_scale": str, 
                                      "weights_shape": str, "weight_bits" : int, 
                                      "weights_flat": str, "bias_bits" : int, 
                                      "bias_flat": str})
    
    # Iterate through csv rows
    min_trunc_guards: list[int] = []
    for i, row in enumerate(df.itertuples()):
        w_bits = int(str(row.weight_bits))
        b_bits = int(str(row.bias_bits))

        # 1. Pack Weights into a single large bit-vector for Header (.vh)
        weights = ast.literal_eval(str(row.weights_flat))
        packed_w = 0
        for idx, val in enumerate(weights):
            # Mask to bit-width to handle negative numbers correctly in hex
            masked_val = int(str(val)) & ((1 << w_bits) - 1)
            packed_w |= (masked_val << (idx * w_bits))

        # 2. Pack Biases into a single large bit-vector for Header (.vh)
        biases = ast.literal_eval(str(row.bias_flat))
        packed_b = 0
        for idx, val in enumerate(biases):
            masked_val = int(str(val)) & ((1 << b_bits) - 1)
            packed_b |= (masked_val << (idx * b_bits))

        w_total_bits = len(weights) * w_bits
        b_total_bits = len(biases) * b_bits

        # Format as Verilog hex literals (e.g. 160'hABC...)
        max_digits = max(len(str(w_total_bits-1)), len(str(b_total_bits-1)))
        layer_i_cfg = config.layers[i].ConvLayer if i < len(config.layers) else config.classifier_config
        if layer_i_cfg._dsp_count == 0:
            sv_lines.append(f"localparam logic signed [{w_total_bits-1:>{max_digits}}:0] LAYER_{i}_WEIGHTS = {w_total_bits}'h{packed_w:x};")
        sv_lines.append(f"localparam logic signed [{b_total_bits-1:>{max_digits}}:0] LAYER_{i}_BIASES  = {b_total_bits}'h{packed_b:x};")
        sv_lines.append("")

        # 3. Determine layer config (needed for shifts and TruncGuard)
        layer_cfg: Any
        if i < len(config.layers):
            layer_cfg = config.layers[i].ConvLayer
        else:
            layer_cfg = config.classifier_config

        # 4. Emit per-layer shift for intermediate 4-bit+ layers (gen_learned_shift path)
        if hasattr(row, 'layer_shift') and str(row.layer_shift) not in ("", "nan", "None"):
            ls = int(float(str(row.layer_shift)))
            sv_lines.append(f"localparam int LAYER_{i}_SHIFT = {ls};  // Per-layer output shift for acc >> LAYER_{i}_SHIFT")
            sv_lines.append("")

        # 5. Compute minimum TruncGuard from actual quantized weights (feature conv layers only)
        if i < len(config.layers):
            if hasattr(row, 'layer_shift') and str(row.layer_shift) not in ("", "nan", "None"):
                hw_shift = int(float(str(row.layer_shift)))
            else:
                # Sign-path layers (out_bits <= 2) don't emit ShiftBits in the renderer;
                # conv_layer.sv defaults ShiftBits to 0.
                hw_shift = 0
            min_tg = compute_min_trunc_guard(
                weights_flat=weights,
                biases=biases,
                in_bits=layer_cfg._in_bits,
                out_bits=layer_cfg._out_bits,
                in_channels=layer_cfg._in_ch,
                out_channels=layer_cfg._out_ch,
                kernel_width=layer_cfg._kernel_width,
                learned_shift=hw_shift,
            )
            min_trunc_guards.append(min_tg)
            sv_lines.append(f"localparam int LAYER_{i}_MIN_TRUNC_GUARD = {min_tg};  // Minimum TruncGuard to cover worst-case acc from actual weights")
            sv_lines.append("")

        # 7. Generate Hex file if layer uses DSPs
        if layer_cfg._dsp_count > 0:
            pack_hex_weights(hex_path, weights, w_bits, layer_cfg._dsp_count, layer_cfg, i)

    with open(sv_path, 'w') as f:
        f.write("\n".join(sv_lines))
    print(f"SystemVerilog export complete! Saved to {sv_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Export QAT Network to Hardware files")
    parser.add_argument("network_in", nargs="?", default=NET_PATH, type=Path)
    parser.add_argument("csv_out", nargs="?", default=DATAPATH / "hardware_weights.csv", type=Path)
    parser.add_argument("sv_out", nargs="?", default=DATAPATH / "hardware_weights.vh", type=Path)
    parser.add_argument("hex_out", nargs="?", default=ROMPATH, type=Path)
    parser.add_argument("--random", action="store_true", help="Generate random weights using q_min_bits instead of loading a network")
    
    args = parser.parse_args()

    if not args.random and not args.network_in.exists():
        print(f"Error: Network file not found at {args.network_in}")
        sys.exit(1)

    # 1. Refresh the CSV from the trained network (or generate random weights)
    export_nn_to_csv(args.network_in, config=NN_CFG, output_csv=args.csv_out, random_weights=args.random)
    export_nn_to_csv(args.network_in, config=NN_CFG, output_csv=args.csv_out, random_weights=args.random)

    # 2. Generate the SystemVerilog header and hex files
    export_csv_to_hex(args.csv_out, args.sv_out, args.hex_out, config=NN_CFG)
    export_csv_to_hex(args.csv_out, args.sv_out, args.hex_out, config=NN_CFG)
    print("Run cnn.py verilog ?")