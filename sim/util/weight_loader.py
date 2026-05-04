# weight_loader.py
import re
import os
from typing import Dict, Any, Tuple
from model.config import ModelConfig
from util.bitwise import unpack_kernel_weights, unpack_weights, unpack_biases

def load_weights_from_vh(vh_path: str, config: ModelConfig) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Parses a .vh file and returns both unpacked arrays and raw integers.
    
    Returns:
        (unpacked_dict, raw_dict)
        unpacked_dict: Dict with keys like 'LAYER_0_WEIGHTS', values are lists.
        raw_dict: Dict with same keys, values are large integers.
    """
    if not os.path.exists(vh_path):
        raise FileNotFoundError(f"Weight file not found: {vh_path}")

    with open(vh_path, 'r') as f:
        content = f.read()

    # Regex to find: localparam logic signed [WIDTH:0] NAME = WIDTH'hHEX;
    # Example: localparam logic signed [ 63:0] LAYER_0_BIASES = 64'hfd04...;
    pattern = r"localparam\s+logic\s+signed\s+\[\s*\d+:\d+\s*\]\s+(\w+)\s*=\s*\d+'h([0-9a-fA-F]+);"
    matches = re.findall(pattern, content)
    
    raw_integers = {name: int(hex_val, 16) for name, hex_val in matches}
    unpacked_dict: Dict[str, Any] = {}

    # Process Conv/Pool Layers (Layers 0 to N-1)
    for i, layer_cfg in enumerate(config.layers):
        cfg = layer_cfg.ConvLayer
        
        # Weights
        w_name = f"LAYER_{i}_WEIGHTS"
        if w_name in raw_integers and cfg._out_ch is not None:
            unpacked_dict[w_name] = unpack_kernel_weights(
                raw_integers[w_name], 
                cfg._weight_bits, 
                cfg._out_ch, 
                cfg._in_ch, 
                cfg._kernel_width
            )
        
        # Biases
        b_name = f"LAYER_{i}_BIASES"
        if b_name in raw_integers and cfg._out_ch is not None:
            unpacked_dict[b_name] = unpack_biases( # type: ignore
                raw_integers[b_name], 
                cfg._bias_bits, 
                cfg._out_ch
            )

    # Process Classifier Layer (The last one)
    c_cfg = config.classifier_config
    c_idx = len(config.layers)
    
    w_name = f"LAYER_{c_idx}_WEIGHTS"
    if w_name in raw_integers:
        unpacked_dict[w_name] = unpack_weights( # type: ignore
            raw_integers[w_name], 
            c_cfg._q_schedule._q_min_bits,
            c_cfg._num_classes, 
            c_cfg._in_ch
        )

    b_name = f"LAYER_{c_idx}_BIASES"
    if b_name in raw_integers:
        unpacked_dict[b_name] = unpack_biases( # type: ignore
            raw_integers[b_name], 
            c_cfg._bias_bits, 
            c_cfg._num_classes
        )

    return unpacked_dict, raw_integers
