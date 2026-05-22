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
        w_scale = max_abs / qmax if max_abs > 0 else 1.0

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
    network.eval()

    all_hardware_data: list[dict[str, Any]] = []
    current_act_scale = 1.0 # Input pixels are binarized {0, 1} or normalized, start at 1.0
    last_w_scale      = 1.0 # Tracks w_scale of the most recent QuantConv2d, needed for LearnedShiftQuantizer

    print("Exporting Fused QAT layers to hardware CSV with scale propagation...")

    # We must iterate through all modules to track activation scales correctly
    # Features followed by Classifier
    all_modules = list(network.features.children()) + list(network.classifier.children())

    # Track the learned classifier output shift separately
    classifier_shift: Optional[int] = None

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
            else:
                qmax = (2 ** (module.bits - 1)) - 1
                if module.clip_val is not None:
                    clip = float(module.clip_val.abs().item())
                    current_act_scale = clip / qmax
                    print(f"  Detected Activation Scaling: {current_act_scale:.6f} (clip={clip:.3f}, qmax={qmax})")
                else:
                    # Handle fixed power-of-two scaling (shift)
                    current_act_scale = 2.0 ** module.shift
                    print(f"  Detected Activation Scaling: {current_act_scale:.6f} (Fixed Shift {module.shift})")

        elif isinstance(module, LearnedShiftQuantizer):
            # Use act_scale (clip_val / qmax) for weight-compensation propagation.
            current_act_scale = module.act_scale
            # Correct shift formula: shift = log2(clip_val / (qmax_out * w_scale))
            qmax_out  = (2 ** module._out_bits) - 1
            clip_abs  = float(module.clip_val.abs().clamp(min=1e-4).item())
            raw_shift = math.log2(clip_abs / (qmax_out * last_w_scale))
            shift_val = max(0, round(raw_shift))

            if all_hardware_data:
                conv_idx = len(all_hardware_data) - 1
                is_last_feature = (conv_idx == len(config.layers) - 1)
                # Always stamp as layer_shift so LAYER_i_SHIFT is emitted to the vh file
                all_hardware_data[-1]["layer_shift"] = shift_val
                if is_last_feature:
                    # Back-propagate into config so RTL render sees the trained value.
                    classifier_shift = shift_val
                    config.classifier_config._shift = classifier_shift
                    all_hardware_data[-1]["classifier_shift"] = classifier_shift
                    print(f"  Learned Classifier Shift: {classifier_shift} (formula gave {raw_shift:.2f}, module.hardware_shift={module.hardware_shift})  act_scale={current_act_scale:.6f}  clip_val={clip_abs:.4f}")
                else:
                    print(f"  Layer {conv_idx} Shift: {shift_val} (formula gave {raw_shift:.2f})  act_scale={current_act_scale:.6f}  clip_val={clip_abs:.4f}")

    # If no LearnedShiftQuantizer was encountered (e.g. random_weights path),
    # fall back to the analytically-derived shift on the last feature conv row
    if all_hardware_data and "classifier_shift" not in all_hardware_data[-2 if len(all_hardware_data) > 1 else -1]:
        feature_row_idx = len(all_hardware_data) - 2  # last feature row, not classifier
        if 0 <= feature_row_idx < len(all_hardware_data):
            all_hardware_data[feature_row_idx]["classifier_shift"] = config.classifier_config._shift

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
        sv_lines.append(f"localparam logic signed [{w_total_bits-1:>{max_digits}}:0] LAYER_{i}_WEIGHTS = {w_total_bits}'h{packed_w:x};")
        sv_lines.append(f"localparam logic signed [{b_total_bits-1:>{max_digits}}:0] LAYER_{i}_BIASES  = {b_total_bits}'h{packed_b:x};")
        sv_lines.append("")

        # 3. Emit per-layer shift for intermediate 4-bit+ layers (gen_learned_shift path)
        if hasattr(row, 'layer_shift') and str(row.layer_shift) not in ("", "nan", "None"):
            ls = int(float(str(row.layer_shift)))
            sv_lines.append(f"localparam int LAYER_{i}_SHIFT = {ls};  // Per-layer output shift for acc >> LAYER_{i}_SHIFT")
            sv_lines.append("")

        # 4. Emit CLASSIFIER_SHIFT for the final feature layer (carries the learned shift)
        if hasattr(row, 'classifier_shift') and str(row.classifier_shift) not in ("", "nan", "None"):
            learned_shift = int(float(str(row.classifier_shift)))
            sv_lines.append(f"localparam int CLASSIFIER_SHIFT = {learned_shift};  // Learned right-shift for acc >> CLASSIFIER_SHIFT")
            sv_lines.append("")
        
        # 4. Generate Hex file if layer uses DSPs
        layer_cfg: Any
        if i < len(config.layers):
            layer_cfg = config.layers[i].ConvLayer
        else:
            layer_cfg = config.classifier_config
            
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

    # 2. Generate the SystemVerilog header and hex files
    export_csv_to_hex(args.csv_out, args.sv_out, args.hex_out, config=NN_CFG)
    print("Run cnn.py render ?")