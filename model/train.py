# train.py
# Bradley Manzo 2026

import kagglehub

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt

from pathlib import Path
import matplotlib.pyplot as plt

from .model import cnn_model, QuantConv2d
from .export import export_model_to_csv

IMG_H, IMG_W = 240, 320
DATA_SPLIT = 0.8
BEGIN_Q_EPOCHS = 20
EPOCHS_PER_LAYER = 15

train_acc_history = []
test_acc_history = []

class LayerConfig:
    def __init__(self, q_start=0, q_epochs=[15,15,15,15,15,15,30], q_max_bits=8, q_min_bits=2):
        self._q_start = q_start
        assert len(q_epochs) == q_max_bits - q_min_bits + 1, "q_epochs must specify epochs for each bit-width quantization step + the final plateau"
        self._epochs_per_bit = q_epochs
        self._q_max_bits = q_max_bits
        self._q_min_bits = q_min_bits

    def total_epochs(self):
        '''Return the total epochs to carry out the final quantization'''
        return self._q_start + sum(self._epochs_per_bit)
    
    def get_target_bits(self, current_epoch):
            '''Returns the target bit-width for a given epoch, or None if quantization hasn't started.'''
            if current_epoch < self._q_start:
                return None
                
            epochs_passed = current_epoch - self._q_start
            accumulated_epochs = 0
            
            # Walk through the schedule to find our current bit-width
            for i, duration in enumerate(self._epochs_per_bit):
                accumulated_epochs += duration
                if epochs_passed < accumulated_epochs:
                    return self._q_max_bits - i
                    
            # If we've passed the final scheduled duration, clamp to min bits
            return self._q_min_bits
    
def pad_to_target(img):
    '''Dataset images are smaller than first conv layer -> pad to 320x240.'''
    # img is PIL here
    w, h = img.size
    pad_left = max((IMG_W - w) // 2, 0)
    pad_top  = max((IMG_H - h) // 2, 0)
    pad_right = max(IMG_W - w - pad_left, 0)
    pad_bottom = max(IMG_H - h - pad_top, 0)
    return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=255)

tfm_train = transforms.Compose([
    transforms.Grayscale(1),
    
    # Pad the image to 320x240 (centers it and fills borders with white: 255)
    transforms.Lambda(pad_to_target),
    
    # Invert colors to match Arducam capture
    transforms.Lambda(lambda img: TF.invert(img)),
    
    # Convert to tensor
    transforms.ToTensor(), 
])

# Rotate by up to 15 degrees and fill borders with white
train_aug = transforms.Compose([
    transforms.RandomRotation(15, fill=1), 
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=1), 
])

# Import hand gesture dataset from Kaggle
path = Path(kagglehub.dataset_download("roobansappani/hand-gesture-recognition"))
dataset_root = path / "HandGesture" / "images"
dataset = datasets.ImageFolder(dataset_root, transform=tfm_train)

# Split dataset into 80% train and 20% test
n = len(dataset) 
n_train = int(DATA_SPLIT * n)
n_test  = n - n_train
train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                 generator=torch.Generator().manual_seed(0))

# Configure DataLoaders on GPU for batch processing
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, 
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, 
                          num_workers=4, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# Import CNN Model and set the optimizer, loss function, and scheduler for training
model = cnn_model(in_ch=1, num_classes=len(dataset.classes)).to(device)

# Establish schedule for progressive quantization of each layer

# t_i, t_b8, t_b7, t_b6, t_b5, t_b4, t_b3, t_b2
schedule = [LayerConfig(20, [15, 15, 15, 15, 15, 20, 30], 8, 2),
            LayerConfig(35, [15, 15, 15, 15, 15, 20, 30], 8, 2),
            LayerConfig(50, [15, 15, 15, 20, 25, 30, 35], 8, 2),
            LayerConfig(65, [15, 15, 15, 20, 25, 30, 40], 8, 2)]

model_layers = model.features.modules()
assert len(schedule) == sum(1 for m in model_layers if isinstance(m, QuantConv2d)), "Schedule must have an entry for each QuantConv2d layer"

EPOCHS = max(cfg.total_epochs() for cfg in schedule) 
BEGIN_Q_EPOCHS = schedule[0]._q_start

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - BEGIN_Q_EPOCHS)

print(f"Training on {device} for {EPOCHS} epochs with progressive quantization starting at epoch {BEGIN_Q_EPOCHS}")
print("\n--- STARTING FULL-PRECISION TRAINING ---")

for epoch in range(EPOCHS):
    if epoch == BEGIN_Q_EPOCHS:
        print("\n--- STARTING PROGRESSIVE QUANTIZATION ---")
        
    # --------------------------------------- SCHEDULING LOGIC ---------------------------------------
    quant_layer_idx = 0
    for module in model.features.modules(): # Safely iterate directly
        if isinstance(module, QuantConv2d):
            # Fetch the target bits directly from your shiny new data structure
            target_bits = schedule[quant_layer_idx].get_target_bits(epoch)
            
            if target_bits is not None:
                # Turning quantization ON for the first time
                if not getattr(module, '_quantize', False): 
                    module._quantize = True
                    module._bits = target_bits
                    print(f"Epoch {epoch:02d}: Layer {quant_layer_idx} quantization ON -> {target_bits}-bit")
                    
                # Dropping to a lower bit-width
                elif getattr(module, '_bits', None) != target_bits:
                    module._bits = target_bits
                    print(f"Epoch {epoch:02d}: Layer {quant_layer_idx} dropping bits -> {target_bits}-bit")
                    
            quant_layer_idx += 1

    # --------------------------------------- TRAINING LOGIC ---------------------------------------
    model.train()
    for x, y in train_loader:
        x = train_aug(x) 
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = loss_fn(logits, y)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        
        # CRITICAL RE-ADDITION: Prevent the weights from exploding on bit drops!
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt.step()

    model.eval()
    train_correct = train_total = 0
    test_correct = test_total = 0
    
    with torch.no_grad():
        # Evaluate Training Set
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Evaluate on unaugmented training images
            pred = model(x).argmax(1)
            train_correct += (pred == y).sum().item()
            train_total += y.numel()
            
        # Evaluate Test Set 
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            test_correct += (pred == y).sum().item()
            test_total += y.numel()

    # Compute accuracies
    train_acc = train_correct / train_total
    test_acc = test_correct / test_total

    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)
    
    # Increment epoch
    scheduler.step()
    
    print(f"epoch {epoch:02d} train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

# --------------------------------------- PLOTTING LOGIC ---------------------------------------
plt.plot(train_acc_history, label="Train Accuracy", linewidth=2)
plt.plot(test_acc_history, label="Test Accuracy", linewidth=2)

colors = ['red', 'orange', 'green', 'blue']
assert len(schedule) == len(colors), "Need a color for each layer in the schedule"

# Find the lowest accuracy on the graph to anchor our text
min_acc = min(min(train_acc_history), min(test_acc_history))

for layer_idx, (cfg, color) in enumerate(zip(schedule, colors)):
    # Offset text vertically so the 4 layers stack neatly instead of overlapping
    y_offset = min_acc + (layer_idx * 0.05) 
    
    # Plot the starting point for this layer
    plt.axvline(cfg._q_start, color=color, linestyle='-', alpha=0.6)
    plt.text(cfg._q_start - 2, y_offset, f'L{layer_idx} Q-Start', rotation=90, color=color, fontsize=8)
    
    # Accumulate the epochs to step forward in time correctly
    accumulated_epochs = cfg._q_start
    for i, duration in enumerate(cfg._epochs_per_bit[:-1]): # Skip the final plateau duration for text labels
        accumulated_epochs += duration
        
        # Calculate the bit-width the layer is dropping TO
        next_bits = max(cfg._q_min_bits, cfg._q_max_bits - i - 1)
        
        # Plot the drop threshold
        plt.axvline(accumulated_epochs, color=color, linestyle='--', alpha=0.3)
        plt.text(accumulated_epochs + 1, y_offset, f'{next_bits}b', rotation=90, color=color, fontsize=8)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy with Quantization Schedule")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

print("\n--- TRAINING COMPLETE ---")
# 1. Save the raw PyTorch model just in case!
model_save_path = "gesture_net_quantized.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Raw PyTorch model saved to: {model_save_path}")

# 2. Extract the folded hardware parameters to CSV
export_model_to_csv(model_save_path, output_csv="hardware_weights.csv")

# 3. Show the graph (Script will pause here until you close the window)
plt.show()