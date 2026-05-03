import torch
import sys
from pathlib import Path

from model.globals    import HAND_GESTURE_CFG, DATAPATH
from model.preprocess import prepare_data, get_transforms

def sample_to_hex(sample_idx=0):
    '''Samples an image from the dataset, applies preprocessing, and returns a Verilog hex string.'''
    
    # 1. Load dataset using standard parameters from HAND_GESTURE_CFG
    dataset_name = "roobansappani/hand-gesture-recognition"
    if HAND_GESTURE_CFG.in_dims.height is None or HAND_GESTURE_CFG.in_dims.width is None or HAND_GESTURE_CFG._in_bits[0] is None:
        print("Error: HAND_GESTURE_CFG is missing input dimensions.")
        return None
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
        return None
    
    # 3. Save Visual Confirmation
    from torchvision.utils import save_image
    img_path = DATAPATH / f"sample_{sample_idx}.png"
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
    
    # 5. Save for Cocotb (JSON)
    # import json
    # json_path = DATAPATH / f"sample_{sample_idx}.json"
    # if label is None:
    #     print(f"Warning: Sample {sample_idx} not found in testloader. Using label 0.")
    #     label = 0

    # sample_data = {
    #     "sample_idx": sample_idx,
    #     "label":  int(label),
    #     "width":  HAND_GESTURE_CFG.in_dims.width,
    #     "height": HAND_GESTURE_CFG.in_dims.height,
    #     "mean":   mean_val,
    #     "pixels": pixels
    # }
    # with open(json_path, 'w') as f:
    #     json.dump(sample_data, f, indent=4)
    # print(f"Saved cocotb-ready data to: {json_path}")

    # # 6. Generate Verilog Hex (Legacy/Injection)
    # packed_val = 0
    # for i, p in enumerate(pixels):
    #     packed_val |= (int(p) << i)
    
    # total_bits = len(pixels)
    # hex_str = f"{total_bits}'h{packed_val:x}"
    
    # # We output the hex to stdout so user can still redirect to .vh if they want
    # print(f"\n// SystemVerilog Injection:")
    # print(f"localparam logic [{total_bits-1}:0] SAMPLE_INPUT = {hex_str};")
    
    # return sample_data

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sample_to_hex(idx)
