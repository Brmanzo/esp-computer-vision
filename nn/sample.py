#!/usr/bin/env python3
# nn.sample.py
# Bradley Manzo 2026

from   pathlib import Path
import sys
from   torchvision.utils import save_image

from nn.globals    import NN_CFG, DATAPATH
from nn.preprocess import prepare_mnist_data

def sample_to_hex(sample_idx: int = 0, path: Path = DATAPATH):
    '''Samples an image from MNIST, applies preprocessing, and returns a Verilog hex string.'''
    if NN_CFG.in_dims.height is None or NN_CFG.in_dims.width is None:
        print("Error: NN_CFG is missing input dimensions.")
        return None, None

    _, test_loader, _ = prepare_mnist_data(
        data_dir   = DATAPATH,
        img_h      = NN_CFG.in_dims.height,
        img_w      = NN_CFG.in_dims.width,
        batch_size = 1,
    )

    img_t, label = None, None
    for i, (batch_img, batch_label) in enumerate(test_loader):
        if i == sample_idx:
            img_t = batch_img[0]
            label = batch_label[0]
            break

    if img_t is None:
        print(f"Error: Sample index {sample_idx} out of range.")
        return None, None

    path.mkdir(parents=True, exist_ok=True)
    img_path = path / f"sample_{sample_idx}.png"
    save_image(img_t, img_path)

    mean_val = img_t.mean().item()
    max_val  = img_t.max().item()
    print(f"--- SAMPLE {sample_idx} (Label: {label}) ---")
    print(f"Stats: Mean={mean_val:.3f}, Max={max_val:.3f}")
    print(f"Saved preprocessed image to: {img_path}")

    # Already binarized by get_transforms; values are exactly 0.0 or 1.0
    pixels = img_t.flatten().int().tolist()

    return pixels, label

def get_sample(sample_idx: int):
    '''Returns preprocessed pixels and label for a specific MNIST test index.'''
    if NN_CFG.in_dims.height is None or NN_CFG.in_dims.width is None:
        print("Error: NN_CFG is missing input dimensions.")
        return None, None

    _, test_loader, _ = prepare_mnist_data(
        data_dir   = DATAPATH,
        img_h      = NN_CFG.in_dims.height,
        img_w      = NN_CFG.in_dims.width,
        batch_size = 1,
    )

    for i, (batch_img, batch_label) in enumerate(test_loader):
        if i == sample_idx:
            img_t = batch_img[0]
            label = int(batch_label[0])
            pixels = img_t.flatten().int().tolist()
            return pixels, label

    return None, None

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sample_to_hex(idx)
