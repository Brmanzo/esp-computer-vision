from typing import Callable, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import kagglehub

from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

def get_transforms(img_h: int, img_w: int, in_bits: int) -> Tuple[Callable, Callable]:
    '''Returns the preprocessing and augmentation transforms for the dataset.'''
    
    def pad_to_target(img):
        '''Finds the gesture and aligns it to the bottom-center of the target frame without scaling.'''
        # 1. Convert to grayscale and numpy
        gray = img.convert('L')
        arr = np.array(gray)
        
        # 2. Binarize to find the hand (Hand=0, BG=255)
        if arr.mean() > 128:
            binary_arr = np.where(arr < 120, 0, 255).astype(np.uint8)
        else:
            binary_arr = np.where(arr > 130, 0, 255).astype(np.uint8)

        # 3. Find bounding box
        coords = np.argwhere(binary_arr == 0)
        if len(coords) == 0:
            return Image.fromarray(binary_arr).resize((img_w, img_h))
            
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        # 4. Crop the hand at its natural size
        hand_crop = Image.fromarray(binary_arr).crop((x0, y0, x1+1, y1+1))
        cw, ch = hand_crop.size
        
        # 5. Place in target frame (Bottom-Aligned, Centered)
        target = Image.new('L', (img_w, img_h), color=255) 
        paste_x = (img_w - cw) // 2
        paste_y = (img_h - ch) # Always bottom align
        target.paste(hand_crop, (paste_x, paste_y))
        
        return target

    tfm_base = transforms.Compose([
        transforms.Grayscale(in_bits),
        transforms.Lambda(pad_to_target),
        # Removed the inversion to keep Black Hand on White Background
        transforms.ToTensor(), 
    ])

    train_aug = transforms.Compose([
        transforms.RandomRotation(15, fill=1), 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=1), 
    ])

    return tfm_base, train_aug

def prepare_data(dataset_download: str, img_h: int, img_w: int, in_bits: int, data_split: float, batch_size: int, max_classes: Optional[int] = None) -> Tuple[DataLoader, DataLoader, int]:
    '''Downloads the dataset and returns DataLoaders along with the number of classes.'''
    path = Path(kagglehub.dataset_download(dataset_download))
    dataset_root = path / "HandGesture" / "images"
    
    tfm_base, _ = get_transforms(img_h, img_w, in_bits)
    dataset = datasets.ImageFolder(dataset_root, transform=tfm_base)
    
    # Constrain classes if requested
    if max_classes is not None:
        target_classes = sorted(dataset.classes)[:max_classes]
        dataset.classes = target_classes
        dataset.class_to_idx = {cls: i for i, cls in enumerate(target_classes)}
        # Filter samples to only include the first max_classes
        dataset.samples = [s for s in dataset.samples if s[1] < max_classes]
        dataset.targets = [s[1] for s in dataset.samples]

    n = len(dataset) 
    n_train = int(data_split * n)
    n_test  = n - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                              num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, len(dataset.classes)
def get_class_names(dataset_name: str = "roobansappani/hand-gesture-recognition") -> list[str]:
    '''Returns the sorted list of class names from the dataset directory.'''
    from pathlib import Path
    import kagglehub
    path = Path(kagglehub.dataset_download(dataset_name))
    dataset_root = path / "HandGesture" / "images"
    return sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
