# nn.inference.py
import argparse
import ast
import math
import numpy as np
import pandas as pd
from   pathlib import Path
import re
import torch
from   typing import Sized, cast

from nn.globals    import DATAPATH, NN_CFG, CLASSES, NET_PATH, prepare_data
from nn.arch       import cnn
from nn.sample     import get_sample

def run_inference(sample_idx: int):
    IMG_H = NN_CFG.in_dims.height
    IMG_W = NN_CFG.in_dims.width
    assert IMG_H is not None and IMG_W is not None
    DATAPATH = Path(__file__).parent / "data"
    NN_PATH = NET_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_names = sorted(CLASSES)
    _, test_loader, _ = prepare_data(DATAPATH, IMG_H, IMG_W, 1)    
    # 3. Load network
    network = cnn(config=NN_CFG)
    
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
        print("\033[92mSUCCESS: Network correctly identified the number!\033[0m")
    else:
        print("\033[91mFAILURE: Network misidentified the number.\033[0m")

def _layer_acc_bits(conv, shift_override: int | None = None, trunc_guard: int = 3) -> int:
    '''Mirror RTL conv_acc_bits(): compute hardware accumulator width for a conv layer.
    shift_override: pass the actual learned shift from hardware_weights.vh; falls back to
    conv._shift (theoretical) only when not supplied.'''
    kA       = conv._kernel_width ** 2
    ib       = conv._in_bits
    wb       = conv._q_schedule._q_min_bits
    ic       = conv._in_ch
    bb       = conv._bias_bits
    sh       = shift_override if shift_override is not None else conv._shift
    ob       = conv._out_bits
    unsigned = 1 if ib > 2 else 0

    if ib == 1:
        max_input = 1
    elif ib == 2 and not unsigned:
        max_input = 1          # ternary {-1,0,+1}
    elif unsigned:
        max_input = (1 << ib) - 1
    else:
        max_input = 1 << (ib - 1)

    max_weight = 1 if wb <= 2 else (1 << (wb - 1))
    worst      = kA * max_input * max_weight * ic
    clog2      = lambda n: 0 if n <= 1 else math.ceil(math.log2(n))
    wc_bits    = clog2(worst + 1) + 1
    wc_bits    = max(wc_bits, bb) + 1
    acc_bits   = min(wc_bits, 32)
    if trunc_guard != 0 and (sh + ob + trunc_guard) < acc_bits:
        acc_bits = sh + ob + trunc_guard
    return acc_bits


def _signed_truncate(acc: np.ndarray, bits: int) -> np.ndarray:
    '''Mask accumulator to a bits-wide two's-complement signed integer,
    simulating hardware modular wrap-around at AccBits.'''
    mask = (1 << bits) - 1
    acc  = acc & mask
    half = 1 << (bits - 1)
    return np.where(acc >= half, acc - (1 << bits), acc)


def _hw_integer_forward(pixels: list, config) -> int:
    '''Core hardware-accurate integer forward pass shared by get_inference() and
    get_inference_from_pixels().  pixels is a flat list of binary {0,1} values.'''

    DATAPATH = Path(__file__).parent / "data"
    CSV_PATH = DATAPATH / "hardware_weights.csv"
    VH_PATH  = DATAPATH / "hardware_weights.vh"

    H = config.in_dims.height
    W = config.in_dims.width

    act = np.array(pixels, dtype=np.int64).reshape(1, H, W)
    act = 2 * act - 1  # binary {0,1} → signed {-1,+1}

    df = pd.read_csv(CSV_PATH)

    with open(VH_PATH, 'r') as f:
        vh_content = f.read()
    m = re.search(r"localparam\s+int\s+CLASSIFIER_SHIFT\s*=\s*(\d+)\s*;", vh_content)
    classifier_shift = int(m.group(1)) if m else config.classifier_config._shift
    layer_shifts: dict[int, int] = {}
    for ms in re.finditer(r"localparam\s+int\s+LAYER_(\d+)_SHIFT\s*=\s*(\d+)\s*;", vh_content):
        layer_shifts[int(ms.group(1))] = int(ms.group(2))
    m_tg = re.search(r"localparam\s+int\s+TRUNC_GUARD\s*=\s*(\d+)\s*;", vh_content)
    global_trunc_guard = int(m_tg.group(1)) if m_tg else 3
    layer_trunc_guards: dict[int, int] = {}
    for mt in re.finditer(r"localparam\s+int\s+LAYER_(\d+)_MIN_TRUNC_GUARD\s*=\s*(\d+)\s*;", vh_content):
        layer_trunc_guards[int(mt.group(1))] = int(mt.group(2))

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
        # Hardware accumulator wraps at AccBits (set by TruncGuard in conv_layer.sv).
        # Use the actual learned shift from the VH file so AccBits matches the bitstream.
        # Sign-path layers (out_bits <= 2) have no LAYER_i_SHIFT entry; RTL ShiftBits defaults to 0.
        hw_shift = layer_shifts.get(i, 0) if i < len(config.layers) - 1 else classifier_shift
        trunc_guard = layer_trunc_guards.get(i, global_trunc_guard)
        acc = _signed_truncate(acc, _layer_acc_bits(conv, shift_override=hw_shift, trunc_guard=trunc_guard))

        if i == len(config.layers) - 1:
            # Last feature layer: ReLU + logical right-shift + unsigned saturation
            shifted = np.maximum(acc, 0) >> classifier_shift
            act = np.clip(shifted, 0, (1 << conv._out_bits) - 1)
        elif conv._out_bits > 2:
            # Intermediate 4-bit+ layer: same gen_learned_shift hardware path as last feature
            ls = layer_shifts.get(i, config.layers[i].ConvLayer._shift)
            shifted = np.maximum(acc, 0) >> ls
            act = np.clip(shifted, 0, (1 << conv._out_bits) - 1)
        else:
            # Ternary (2-bit) / binary (1-bit): sign → {-1, 0, +1}
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
    pixels, _ = get_sample(sample_idx)
    if pixels is None:
        raise ValueError(f"Could not load sample {sample_idx}")
    return _hw_integer_forward(pixels, NN_CFG)


def get_inference_from_pixels(pixels: list, config=None) -> int:
    '''Hardware-accurate inference on an explicit flat list of binary {0,1} pixels.'''
    return _hw_integer_forward(pixels, config or NN_CFG)


def hw_eval(n_trials: int = 100) -> float:
    '''Run hardware-accurate integer inference on n_trials test samples and report accuracy.'''

    DATAPATH = Path(__file__).parent / "data"
    IMG_H, IMG_W = NN_CFG.in_dims.height, NN_CFG.in_dims.width
    assert IMG_H is not None and IMG_W is not None, "Input dimensions must be specified in config"
    _, test_loader, _ = prepare_mnist_data(DATAPATH, IMG_H, IMG_W, 1)

    class_names = sorted(MNIST_CLASSES)
    correct = 0
    total   = 0
    per_class_correct = {name: 0 for name in class_names}
    per_class_total   = {name: 0 for name in class_names}

    for img_t, label_t in test_loader:
        if total >= n_trials:
            break
        label  = int(label_t[0])
        pixels = (img_t[0].flatten() > 0.5).int().tolist()
        pred   = _hw_integer_forward(pixels, NN_CFG)

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
