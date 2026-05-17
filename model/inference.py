# model.inference.py
import torch
import sys
import kagglehub
from pathlib import Path
from PIL import Image

from model.globals import HAND_GESTURE_CFG
from model.model import cnn_model
from model.preprocess import get_transforms, prepare_data

def run_inference(sample_idx: int):
    # 1. Setup Constants
    IMG_H, IMG_W = 240, 320
    IN_BITS = 1
    DATAPATH = Path(__file__).parent / "data"
    MODEL_PATH = DATAPATH / "gesture_net_quantized.pth"
    dataset_name = "roobansappani/hand-gesture-recognition"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Data and Metadata
    # We need the class names to interpret the prediction
    # We'll get them directly from the directory structure for maximum robustness
    path = Path(kagglehub.dataset_download(dataset_name))
    dataset_root = path / "HandGesture" / "images"
    class_names = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
    
    # Still need the loader to get the images
    _, test_loader, _ = prepare_data(dataset_name, IMG_H, IMG_W, IN_BITS, 0.8, 32)
    
    # 3. Load Model
    model = cnn_model(config=HAND_GESTURE_CFG)
    
    if not MODEL_PATH.exists():
        print(f"Error: Model weights not found at {MODEL_PATH}. Run training first.")
        return

    print(f"Loading weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 4. Get the exact same sample from the dataset
    # (Matches model.sample logic)
    dataset = test_loader.dataset
    
    # Cast to Sized for type checker (Subsets always have __len__)
    from typing import Sized, cast
    dataset_len = len(cast(Sized, dataset))
    
    if sample_idx >= dataset_len:
        print(f"Error: Sample index {sample_idx} out of range (Test set size: {dataset_len})")
        return
        
    img_pil, label_raw = dataset[sample_idx]
    label: int = int(label_raw)
    
    # img_pil is already preprocessed by dataset transforms (including our grounding/centering)
    # But we need to add the batch dimension for the model
    input_tensor = img_pil.unsqueeze(0).to(device)

    # 5. Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx: int = int(torch.argmax(probs, dim=1).item())
        confidence: float = float(probs[0][pred_idx].item())

    # 6. Report
    print(f"\n--- INFERENCE RESULT FOR SAMPLE {sample_idx} ---")
    print(f"Ground Truth: {class_names[label]} (Label: {label})")
    print(f"Prediction:   {class_names[pred_idx]} (Label: {pred_idx})")
    print(f"Confidence:   {confidence:.2%}")
    
    if pred_idx == label:
        print("\033[92mSUCCESS: Model correctly identified the gesture!\033[0m")
    else:
        print("\033[91mFAILURE: Model misidentified the gesture.\033[0m")

def _hw_integer_forward(pixels: list, config) -> int:
    '''Core hardware-accurate integer forward pass shared by get_inference() and
    get_inference_from_pixels().  pixels is a flat list of binary {0,1} values.'''
    import re
    import ast
    import numpy as np
    import pandas as pd

    DATAPATH = Path(__file__).parent / "data"
    CSV_PATH = DATAPATH / "hardware_weights.csv"
    VH_PATH  = DATAPATH / "hardware_weights.vh"

    H = config.in_dims.height
    W = config.in_dims.width

    act = np.array(pixels, dtype=np.int64).reshape(1, H, W)
    act = 2 * act - 1  # binary {0,1} → signed {-1,+1}

    df = pd.read_csv(CSV_PATH)

    with open(VH_PATH, 'r') as f:
        m = re.search(r"localparam\s+int\s+CLASSIFIER_SHIFT\s*=\s*(\d+)\s*;", f.read())
    classifier_shift = int(m.group(1)) if m else config.classifier_config._shift

    def _conv(x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int, pad_value: int = 0) -> np.ndarray:
        if pad > 0:
            x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), constant_values=pad_value)
        _, Hp, Wp = x.shape
        OC, _, kH, kW = w.shape
        OH, OW = Hp - kH + 1, Wp - kW + 1
        out = np.zeros((OC, OH, OW), dtype=np.int64)
        for kh in range(kH):
            for kw_i in range(kW):
                out += np.einsum('oi,ixy->oxy',
                                 w[:, :, kh, kw_i],
                                 x[:, kh:kh + OH, kw_i:kw_i + OW],
                                 optimize=True)
        return out + b[:, None, None]

    for i, layer_cfg in enumerate(config.layers):
        conv = layer_cfg.ConvLayer
        row  = df.iloc[i]
        oc, ic, kw = conv._out_ch, conv._in_ch, conv._kernel_width

        w = np.array(ast.literal_eval(str(row.weights_flat)), dtype=np.int64).reshape(oc, ic, kw, kw)
        b = np.array(ast.literal_eval(str(row.bias_flat)),    dtype=np.int64)

        # Binary input layer (InBits=1): hardware pads with 1'b0, which mac.sv maps to -weight.
        # Ternary layers (InBits=2): hardware pads with 2'b00 = ternary 0, contributing 0.
        pad_value = -1 if i == 0 else 0
        acc = _conv(act, w, b, conv._padding, pad_value)

        if i == len(config.layers) - 1:
            # Last feature layer: ReLU + logical right-shift + unsigned saturation
            shifted = np.maximum(acc, 0) >> classifier_shift
            act = np.clip(shifted, 0, (1 << conv._out_bits) - 1)
        else:
            # Ternary activation matching hardware output_encoder: sign(acc) → {-1, 0, +1}
            act = np.sign(acc)

        pool = layer_cfg.PoolLayer
        if pool is not None:
            kp = pool._kernel_width
            _, H2, W2 = act.shape
            OH2 = (H2 - kp) // kp + 1
            OW2 = (W2 - kp) // kp + 1
            pooled = np.zeros((act.shape[0], OH2, OW2), dtype=np.int64)
            for r in range(OH2):
                for c in range(OW2):
                    pooled[:, r, c] = np.max(act[:, r*kp:r*kp+kp, c*kp:c*kp+kp], axis=(1, 2))
            act = pooled

    act_gmax = np.max(act, axis=(1, 2))

    c_idx = len(config.layers)
    row   = df.iloc[c_idx]
    nc    = config.classifier_config._num_classes
    ic_c  = config.classifier_config._in_ch

    w_cls = np.array(ast.literal_eval(str(row.weights_flat)), dtype=np.int64).reshape(nc, ic_c)
    b_cls = np.array(ast.literal_eval(str(row.bias_flat)),    dtype=np.int64)

    logits = w_cls @ act_gmax + b_cls

    # Hardware classifier_layer passes raw signed 32-bit logits to comparator_tree.
    # output_encoder takes the gen_full_out (identity) path because OutBits==LinearBits==32.
    # No ReLU or saturation is applied before argmax.
    return int(np.argmax(logits))


def get_inference(sample_idx: int) -> int:
    '''Hardware-accurate inference on a dataset sample (index into test set).'''
    from model.sample import get_sample
    pixels, _ = get_sample(sample_idx)
    if pixels is None:
        raise ValueError(f"Could not load sample {sample_idx}")
    return _hw_integer_forward(pixels, HAND_GESTURE_CFG)


def get_inference_from_pixels(pixels: list, config=None) -> int:
    '''Hardware-accurate inference on an explicit flat list of binary {0,1} pixels.'''
    return _hw_integer_forward(pixels, config or HAND_GESTURE_CFG)

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_inference(idx)
