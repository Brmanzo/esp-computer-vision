# export.py
import torch
import pandas as pd
import numpy as np
from .model import cnn_model, QuantConv2d
from .config import ModelConfig

def get_hardware_weights(fused_layer: QuantConv2d):
    """
    Extracts the folded integer weights and biases directly from the QAT layer.
    """
    # 1. Recreate the exact fold that happened during training
    bn_scale = fused_layer.bn_weight / torch.sqrt(fused_layer.running_var + fused_layer.eps)
    folded_w = fused_layer.weight * bn_scale.view(-1, 1, 1, 1)
    folded_b = fused_layer.bn_bias - (fused_layer.running_mean * bn_scale)

    # 2. Retrieve the target bits directly from the layer
    target_bits = fused_layer._bits

    # 3. Quantize to extract the exact integer arrays the network learned
    qmin = -2**(target_bits-1)
    qmax = 2**(target_bits-1) - 1

    max_abs = folded_w.abs().max()
    w_scale = max_abs / qmax if max_abs > 0 else 1.0
    
    # Extract integer weights
    q_weights = torch.round(folded_w / w_scale)
    q_weights = torch.clip(q_weights, qmin, qmax)
    
    # Extract integer bias (using the exact same scale)
    hw_bias = torch.round(folded_b / w_scale)
    
    return q_weights, hw_bias

def export_model_to_csv(model_path: str, config: ModelConfig, output_csv="hardware_weights.csv"):
    device = "cpu"

    # Initialize model using the new config architecture
    model = cnn_model(config=config).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    all_hardware_data = []

    print("Exporting Fused QAT layers to hardware CSV...")

    # Dynamically grab all Fused modules so we don't have to guess indices!
    fused_layers = []
    for name, module in model.named_modules():
        if isinstance(module, QuantConv2d):
            fused_layers.append((name, module))

    for name, fused_layer in fused_layers:
        bits = fused_layer._bits
        print(f"  Processing {name} at {bits}-bits...")
        
        # Extract folded weights and biases
        w_int_tensor, b_hw = get_hardware_weights(fused_layer)

        # Convert to numpy
        w_int = w_int_tensor.to(torch.int8).cpu().numpy()
        b_int = b_hw.to(torch.int32).cpu().numpy()

        all_hardware_data.append({
            "layer_name": name,
            "weights_shape": str(tuple(w_int.shape)),
            "weights_flat": w_int.flatten().tolist(),
            "bias_flat": b_int.flatten().tolist(), # Now entirely integers!
        })

    df = pd.DataFrame(all_hardware_data)
    df.to_csv(output_csv, index=False)
    print(f"\nHardware export complete! Saved to {output_csv}")