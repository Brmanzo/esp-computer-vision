import torch
import sys
from pathlib import Path

from model.globals    import HAND_GESTURE_CFG, DATAPATH
from model.preprocess import prepare_data, get_transforms

def sample_to_hex(sample_idx:int=0, path:Path=DATAPATH):
    '''Samples an image from the dataset, applies preprocessing, and returns a Verilog hex string.'''
    
    # 1. Load dataset using standard parameters from HAND_GESTURE_CFG
    dataset_name = "roobansappani/hand-gesture-recognition"
    if HAND_GESTURE_CFG.in_dims.height is None or HAND_GESTURE_CFG.in_dims.width is None or HAND_GESTURE_CFG._in_bits[0] is None:
        print("Error: HAND_GESTURE_CFG is missing input dimensions.")
        return None, None
    train_loader, test_loader, num_classes = prepare_data(
        dataset_name, 
        img_h      = HAND_GESTURE_CFG.in_dims.height,
        img_w      = HAND_GESTURE_CFG.in_dims.width,
        in_bits    = HAND_GESTURE_CFG._in_bits[0],
        data_split = 0.8,
        batch_size = 1
    )
    
    # 2. Grab the sample from the test loader
    # We iterate to the requested index
    img_t, label = None, None
    for i, (batch_img, batch_label) in enumerate(test_loader):
        if i == sample_idx:
            img_t = batch_img[0] # Loader returns [Batch, Channels, H, W]
            label = batch_label[0]
            break
    
    if img_t is None:
        print(f"Error: Sample index {sample_idx} out of range.")
        return None, None
    
    # 3. Save Visual Confirmation
    from torchvision.utils import save_image
    img_path = path / f"sample_{sample_idx}.png"
    save_image(img_t, img_path)
    
    # Calculate stats to see if we should be inverting
    mean_val = img_t.mean().item()
    max_val = img_t.max().item()
    print(f"--- SAMPLE {sample_idx} (Label: {label}) ---")
    print(f"Stats: Mean={mean_val:.3f}, Max={max_val:.3f}")
    print(f"Saved preprocessed image to: {img_path}")

    # 4. Convert to Hardware Integer Format
    # Assuming InBits=1, the transform already produced {0, 1}
    # If Mean > 0.5, the background is likely white (1), which might be wrong for hardware
    pixels = (img_t.flatten() > 0.5).int().tolist()
    
    return pixels, label

def get_sample(sample_idx: int):
    '''Returns preprocessed pixels and label for a specific dataset index.'''
    dataset_name = "roobansappani/hand-gesture-recognition"
    if HAND_GESTURE_CFG.in_dims.height is None or HAND_GESTURE_CFG.in_dims.width is None or HAND_GESTURE_CFG._in_bits[0] is None:
        print("Error: HAND_GESTURE_CFG is missing input dimensions.")
        return None, None
    train_loader, test_loader, num_classes = prepare_data(
        dataset_name, 
        img_h      = HAND_GESTURE_CFG.in_dims.height,
        img_w      = HAND_GESTURE_CFG.in_dims.width,
        in_bits    = HAND_GESTURE_CFG._in_bits[0],
        data_split = 0.8,
        batch_size = 1
    )
    
    for i, (batch_img, batch_label) in enumerate(test_loader):
        if i == sample_idx:
            img_t = batch_img[0]
            label = int(batch_label[0])
            pixels = (img_t.flatten() > 0.5).int().tolist()
            return pixels, label
            
    return None, None

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sample_to_hex(idx)
