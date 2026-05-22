#!/usr/bin/env python3
# nn.globals.py

from pathlib import Path

from nn.tasks.mnist.mnist     import get_nn_cfg as _get_mnist_cfg, MNIST_NET_PATH, MNIST_CLASSES
from nn.tasks.mnist.preprocess import prepare_mnist_data as _prepare_mnist_data, get_transforms as _get_mnist_transforms

# Default paths matching the training pipeline
DATAPATH = Path("nn") / "data"
ROMPATH  = DATAPATH / "roms" / "hex"

BAUD = 115200  # 12 MHz / (prescale=13 * 8) ≈ 115385 baud (0.16% error)
BRAM_COUNT = 30 - 1 # Subtract 1 for the Skid Buffer BRAM on deframer
DSP_COUNT  = 8

# MNIST Task
NET_PATH       = MNIST_NET_PATH
CLASSES        = MNIST_CLASSES
NN_CFG         = _get_mnist_cfg()
prepare_data   = _prepare_mnist_data
get_transforms = _get_mnist_transforms
