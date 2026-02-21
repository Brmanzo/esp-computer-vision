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

from model import cnn_model, QuantConv2d

IMG_H, IMG_W = 240, 320
DATA_SPLIT = 0.8
EPOCHS = 70
BEGIN_Q_EPOCHS = 20
train_acc_history = []
test_acc_history = []

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
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import CNN Model and set the optimizer, loss function, and scheduler for training
model = cnn_model(in_ch=1, num_classes=len(dataset.classes)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

for epoch in range(EPOCHS):
    # Progressively enable Quantization-Aware Training (QAT)
    if epoch >= BEGIN_Q_EPOCHS:
        
        # Reset the scheduler at the start of the ramp
        if epoch == BEGIN_Q_EPOCHS:
            print("\n--- STARTING PROGRESSIVE 2-BIT QUANTIZATION ---")
            # Create a fresh scheduler for the remaining epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - BEGIN_Q_EPOCHS)
            
        # Progressively flip the layers
        quant_layer_idx = 0
        for module in model.modules():
            if isinstance(module, QuantConv2d):
                # Turn on Layer 0 at BEGIN_Q_EPOCHS, Layer 1 at BEGIN_Q_EPOCHS+1, etc.
                if epoch >= BEGIN_Q_EPOCHS + quant_layer_idx:
                    # Optional: Only print if it's currently False so it doesn't spam your console
                    if not getattr(module, '_quantize', False): 
                        print(f"Epoch {epoch}: Enabling quantization for QuantConv2d layer {quant_layer_idx}")
                        module._quantize = True
                        
                quant_layer_idx += 1 # Only count the QuantConv2d layers
        
    model.train()
    for x, y in train_loader:
        
        # Apply augmentations during training to improve generalization
        x = train_aug(x) 
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = loss_fn(logits, y)
        
        opt.zero_grad()
        loss.backward()
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

# Plot training and test accuracy over epochs
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(test_acc_history, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy over Epochs")
plt.legend()
plt.grid()
plt.show()