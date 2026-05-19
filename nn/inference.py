# nn.inference.py
import torch
import sys
import kagglehub
from pathlib import Path
from PIL import Image

from nn.globals import HAND_GESTURE_CFG, GESTURE_CLASSES
from nn.arch import cnn
from nn.preprocess import get_transforms, prepare_data

def run_inference(sample_idx: int):
    # 1. Setup Constants
    IMG_H, IMG_W = 240, 320
    IN_BITS = 1
    DATAPATH = Path(__file__).parent / "data"
    NN_PATH = DATAPATH / "gesture_net_quantized.pth"
    dataset_name = "roobansappani/hand-gesture-recognition"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Data and Metadata
    class_names = sorted(GESTURE_CLASSES)  # okay=0, paper=1, peace=2, up=3

    # Still need the loader to get the images
    _, test_loader, _ = prepare_data(dataset_name, IMG_H, IMG_W, IN_BITS, 0.8, 32, target_classes=GESTURE_CLASSES)
    
    # 3. Load network
    network = cnn(config=HAND_GESTURE_CFG)
    
    if not NN_PATH.exists():
        print(f"Error: network weights not found at {NN_PATH}. Run training first.")
        return

    print(f"Loading weights from {NN_PATH}...")
    network.load_state_dict(torch.load(NN_PATH, map_location=device))
    network.to(device)
    network.eval()

    # 4. Get the exact same sample from the dataset
    # (Matches nn.sample logic)
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
    # But we need to add the batch dimension for the network
    input_tensor = img_pil.unsqueeze(0).to(device)

    # 5. Inference
    with torch.no_grad():
        logits = network(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx: int = int(torch.argmax(probs, dim=1).item())
        confidence: float = float(probs[0][pred_idx].item())

    # 6. Report
    print(f"\n--- INFERENCE RESULT FOR SAMPLE {sample_idx} ---")
    print(f"Ground Truth: {class_names[label]} (Label: {label})")
    print(f"Prediction:   {class_names[pred_idx]} (Label: {pred_idx})")
    print(f"Confidence:   {confidence:.2%}")
    
    if pred_idx == label:
        print("\033[92mSUCCESS: Network correctly identified the gesture!\033[0m")
    else:
        print("\033[91mFAILURE: Network misidentified the gesture.\033[0m")

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
    from nn.sample import get_sample
    pixels, _ = get_sample(sample_idx)
    if pixels is None:
        raise ValueError(f"Could not load sample {sample_idx}")
    return _hw_integer_forward(pixels, HAND_GESTURE_CFG)


def get_inference_from_pixels(pixels: list, config=None) -> int:
    '''Hardware-accurate inference on an explicit flat list of binary {0,1} pixels.'''
    return _hw_integer_forward(pixels, config or HAND_GESTURE_CFG)


def hw_eval(n_trials: int = 100) -> float:
    '''Run hardware-accurate integer inference on n_trials test samples and report accuracy.'''
    import numpy as np
    from nn.preprocess import get_transforms
    from nn.globals import GESTURE_CLASSES

    dataset_name = "roobansappani/hand-gesture-recognition"
    IMG_H, IMG_W = HAND_GESTURE_CFG.in_dims.height, HAND_GESTURE_CFG.in_dims.width
    IN_BITS = HAND_GESTURE_CFG._in_bits[0]
    assert IMG_H is not None and IMG_W is not None, "Input dimensions must be specified in config"
    _, test_loader, _ = prepare_data(dataset_name, IMG_H, IMG_W, IN_BITS, 0.8, 1,
                                     target_classes=GESTURE_CLASSES)

    class_names = sorted(GESTURE_CLASSES)
    correct = 0
    total   = 0
    per_class_correct = {name: 0 for name in class_names}
    per_class_total   = {name: 0 for name in class_names}

    for img_t, label_t in test_loader:
        if total >= n_trials:
            break
        label  = int(label_t[0])
        pixels = (img_t[0].flatten() > 0.5).int().tolist()
        pred   = _hw_integer_forward(pixels, HAND_GESTURE_CFG)

        name = class_names[label]
        per_class_total[name]   += 1
        per_class_correct[name] += int(pred == label)
        correct += int(pred == label)
        total   += 1

        print(f"  [{total:>4}/{n_trials}]  gt={class_names[label]:<12} pred={class_names[pred]:<12} {'✓' if pred == label else '✗'}")

    acc = correct / total if total else 0.0
    print(f"\nHW accuracy over {total} trials: {acc:.1%}")
    print("\nPer-class accuracy:")
    for name in class_names:
        n = per_class_total[name]
        c = per_class_correct[name]
        print(f"  {name:<12} {c}/{n}  ({c/n:.1%})" if n else f"  {name:<12} no samples")
    return acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_infer = sub.add_parser("infer", help="Single-sample float inference")
    p_infer.add_argument("idx", type=int)

    p_eval = sub.add_parser("hw-eval", help="Batch hardware-accurate accuracy eval")
    p_eval.add_argument("--trials", type=int, default=100)

    args = parser.parse_args()

    if args.cmd == "hw-eval":
        hw_eval(args.trials)
        print("Run cnn.py export ?")
        print("Run cnn.py render ?")
    else:
        idx = args.idx if args.cmd == "infer" else 10
        run_inference(idx)
