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

def get_inference(sample_idx: int):
    '''Performs PyTorch inference on a specific sample and returns the predicted class index.'''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATAPATH = Path(__file__).parent / "data"
    MODEL_PATH = DATAPATH / "gesture_net_quantized.pth"
    dataset_name = "roobansappani/hand-gesture-recognition"

    _, test_loader, _ = prepare_data(dataset_name, 240, 320, 1, 0.8, 1)
    model = cnn_model(config=HAND_GESTURE_CFG)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    dataset = test_loader.dataset
    img_pil, _ = dataset[sample_idx]
    input_tensor = img_pil.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        pred_idx = int(torch.argmax(logits, dim=1).item())
    
    return pred_idx

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_inference(idx)
