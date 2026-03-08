# export.py
import torch
import pandas as pd
import numpy as np
from typing import cast
from .model import cnn_model, QuantConv2d

def get_quantized_data(conv_layer: QuantConv2d, target_bits: int):
    """
    Extracts the actual integer weights (q), fake-quantized floats (w_q), and scale.
    """
    w = conv_layer.weight.detach()
    qmin = -2**(target_bits-1)
    qmax = 2**(target_bits-1) - 1

    max_abs = w.abs().max()
    scale = max_abs / qmax if max_abs > 0 else 1.0

    # Extract the raw integers
    q = torch.round(w / scale)
    q = torch.clip(q, qmin, qmax)
    
    # Calculate the fake-quantized weights
    w_q = q * scale

    return q, w_q, scale

def fold_batchnorm_into_conv(conv_layer: QuantConv2d, bn_layer: torch.nn.BatchNorm2d, target_bits: int):
    q, w_q, w_scale = get_quantized_data(conv_layer, target_bits)

    gamma = bn_layer.weight.detach()
    beta = bn_layer.bias.detach()
    if bn_layer.running_mean is not None and bn_layer.running_var is not None:
        mean = bn_layer.running_mean.detach()
        var = bn_layer.running_var.detach()
    else:
        return q, torch.zeros_like(beta)  # If no running stats, we can't fold, so return zero bias
    eps = bn_layer.eps

    bn_scale = gamma / torch.sqrt(var + eps)

    if conv_layer.bias is not None:
        conv_bias = conv_layer.bias.detach()
    else:
        conv_bias = torch.zeros_like(mean)

    # 1. Calculate the standard folded bias (float)
    folded_bias = (conv_bias - mean) * bn_scale + beta
    
    # 2. HARDWARE TRICK: Scale the bias into the integer domain!
    # Because your hardware does (Input * q) + Bias, we divide the float bias 
    # by the total mathematical scale so you can add it directly to your integer accumulator.
    total_scale = w_scale * bn_scale
    hw_bias = folded_bias / total_scale

    # Return the pure integer weights 'q', and the scaled-down bias
    return q, hw_bias

def export_model_to_csv(model_path, num_classes, output_csv="hardware_weights.csv"):
    device = "cpu"

    model = cnn_model(in_ch=1, num_classes=num_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    all_hardware_data = []

    print("Exporting layers with BatchNorm folding...")

    # Notice the updated indices to account for the new BN and Activation layers!
    # Format: (Name, Conv_Layer, BN_Layer, Target_Bits)
    conv_layers = [
        ("Block0", cast(QuantConv2d, model.features[0]), cast(torch.nn.BatchNorm2d, model.features[1]), 2),
        ("Block1", cast(QuantConv2d, model.features[4]), cast(torch.nn.BatchNorm2d, model.features[5]), 2),
        ("Block2", cast(QuantConv2d, model.features[8]), cast(torch.nn.BatchNorm2d, model.features[9]), 2),
        ("Block3", cast(QuantConv2d, model.features[12]), cast(torch.nn.BatchNorm2d, model.features[13]), 3),
        ("Classifier", cast(QuantConv2d, model.classifier[1]), cast(torch.nn.BatchNorm2d, model.classifier[2]), 8),
    ]

    for name, conv, bn, bits in conv_layers:
        print(f"  Processing {name} at {bits}-bits...")
        
        # Extract folded weights and biases
        w_int_tensor, b_hw = fold_batchnorm_into_conv(conv, bn, target_bits=bits)

        # Convert to numpy
        w_int = w_int_tensor.to(torch.int8).cpu().numpy()
        
        # Round the scaled bias to the nearest integer for your SystemVerilog module
        b_int = torch.round(b_hw).to(torch.int32).cpu().numpy()

        all_hardware_data.append({
            "layer_name": name,
            "weights_shape": str(tuple(w_int.shape)),
            "weights_flat": w_int.flatten().tolist(),
            "bias_flat": b_int.flatten().tolist(), # Now entirely integers!
        })

    df = pd.DataFrame(all_hardware_data)
    df.to_csv(output_csv, index=False)
    print(f"\nHardware export complete! Saved to {output_csv}")