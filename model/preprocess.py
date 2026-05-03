from typing import Callable, Tuple

import torch
import kagglehub

from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

def get_transforms(img_h: int, img_w: int, in_bits: int) -> Tuple[Callable, Callable]:
    '''Returns the preprocessing and augmentation transforms for the dataset.'''
    
    def pad_to_target(img):
        '''Dataset images are smaller than first conv layer -> pad to target dimensions.'''
        w, h = img.size
        pad_left = max((img_w - w) // 2, 0)
        pad_top  = max((img_h - h) // 2, 0)
        pad_right = max(img_w - w - pad_left, 0)
        pad_bottom = max(img_h - h - pad_top, 0)
        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=255)

    tfm_base = transforms.Compose([
        transforms.Grayscale(in_bits),
        transforms.Lambda(pad_to_target),
        transforms.Lambda(lambda img: TF.invert(img)),
        transforms.ToTensor(), 
    ])

    train_aug = transforms.Compose([
        transforms.RandomRotation(15, fill=1), 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=1), 
    ])

    return tfm_base, train_aug

def prepare_data(dataset_download: str, img_h: int, img_w: int, in_bits: int, data_split: float, batch_size: int) -> Tuple[DataLoader, DataLoader, int]:
    '''Downloads the dataset and returns DataLoaders along with the number of classes.'''
    path = Path(kagglehub.dataset_download(dataset_download))
    dataset_root = path / "HandGesture" / "images"
    
    tfm_base, _ = get_transforms(img_h, img_w, in_bits)
    dataset = datasets.ImageFolder(dataset_root, transform=tfm_base)
    
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