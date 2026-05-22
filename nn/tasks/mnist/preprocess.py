#!/usr/bin/env python3
# nn.tasks.mnist.preprocess.py
# Bradley Manzo 2026

from   pathlib import Path
from   typing import Callable, Tuple

from torch.utils.data       import DataLoader
from torchvision            import datasets, transforms

def get_transforms(img_h: int, img_w: int) -> Tuple[Callable, Callable]:
    '''Returns train and test transforms for MNIST.'''
    train_tfm = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.RandomRotation(15, fill=0),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=0),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),
    ])

    return train_tfm, test_tfm

def prepare_mnist_data(data_dir: str | Path, img_h: int, img_w: int, batch_size: int) -> Tuple[DataLoader, DataLoader, int]:
    '''Downloads MNIST via torchvision and returns DataLoaders using the standard 60K/10K split.'''
    train_tfm, test_tfm = get_transforms(img_h, img_w)

    train_ds = datasets.MNIST(root=str(data_dir), train=True,  download=True, transform=train_tfm)
    test_ds  = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=test_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, 10