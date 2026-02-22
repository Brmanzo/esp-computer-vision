# export_weights.py
# Run this after saving your trained model to extract folded hardware parameters

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from model import cnn_model, QuantConv2d # Import your specific architecture

def fold_batchnorm_into_conv(conv_layer, bn_layer):
    """
    Hardware Folding:
    Mathematically merges the BatchNorm scaling and shifting into the 
    convolution weights and bias so the hardware only has to do one operation.
    """
    w_q = conv_layer.weight # We will assume weights are already quantized to [-1,0,1]
    
    # BatchNorm parameters
    gamma = bn_layer.weight
    beta = bn_layer.bias
    mean = bn_layer.running_mean
    var = bn_layer.running_var
    eps = bn_layer.eps

    # Calculate the folding scale factor
    scale = gamma / torch.sqrt(var + eps)
    
    # 1. Fold scale into the bias
    # If the conv layer didn't have a bias (yours don't), we assume it's 0
    conv_bias = conv_layer.bias if conv_layer.bias is not None else torch.zeros_like(mean)
    folded_bias = (conv_bias - mean) * scale + beta
    
    # 2. Fold scale into the weights
    # Note: We reshape scale so it broadcasts across the output channels correctly
    folded_weights = w_q * scale.view(-1, 1, 1, 1)

    return folded_weights, folded_bias

def export_model_to_csv(model_path, output_csv="hardware_weights.csv"):
    device = "cpu" # Hardware export should happen on CPU
    
    # Load your trained model structure (match the wider architecture!)
    model = cnn_model(in_ch=1, num_classes=5).to(device) # Make sure num_classes matches Kaggle!
    
    # Load the trained parameters
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_hardware_data = []

    # Iterate through the Sequential blocks looking for Conv->BatchNorm pairs
    print("Folding layers for hardware extraction...")
    
    # We will manually pair them up based on your architecture
    layer_pairs = [
        ("Block0", model.features[0], model.features[1]),
        ("Block1", model.features[4], model.features[5]),
        ("Block2", model.features[8], model.features[9]),
        ("Block3", model.features[12], model.features[13]),
        ("Classifier", model.classifier[1], model.classifier[2]) # The 8-bit layer
    ]

    for name, conv, bn in layer_pairs:
        print(f"  Processing {name}...")
        
        # Calculate the folded math
        folded_w, folded_b = fold_batchnorm_into_conv(conv, bn)
        
        # Since the weights are already trained as [-1, 0, 1], we round to ensure 
        # perfect integers for the CSV export.
        folded_w_int = torch.round(folded_w).to(torch.int8).numpy()
        
        # We leave the folded bias as a float for now. You might need to quantize 
        # this bias to an integer later depending on your exact hardware spec.
        folded_b_float = folded_b.detach().numpy() 

        # Flatten and save
        all_hardware_data.append({
            "layer_name": name,
            "weights_shape": str(folded_w_int.shape),
            "weights_flat": folded_w_int.flatten().tolist(),
            "bias_flat": folded_b_float.tolist()
        })

    # Save to CSV
    df = pd.DataFrame(all_hardware_data)
    df.to_csv(output_csv, index=False)
    print(f"\nHardware export complete! Saved to {output_csv}")

if __name__ == "__main__":
    # You will need to add a line to your train.py to save the model at the end:
    # torch.save(model.state_dict(), "final_model.pth")
    
    # export_model_to_csv("final_model.pth")
    print("Ready to export once training is done!")