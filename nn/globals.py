#!/usr/bin/env python3
# nn.globals.py

from pathlib import Path

from nn.tasks.hand_gesture.hand_gesture import get_hand_gesture_cfg, GESTURE_CLASSES, GESTURE_NET_PATH
from nn.tasks.hand_gesture.preprocess  import prepare_data as _prepare_data, get_transforms as _get_transforms
from nn.tasks.mnist.mnist import get_nn_cfg as _get_mnist_cfg, MNIST_NET_PATH, MNIST_CLASSES

# Default paths matching the training pipeline
DATAPATH = Path("nn") / "data"
ROMPATH  = DATAPATH / "roms" / "hex"

BAUD = 115200  # 12 MHz / (prescale=13 * 8) ≈ 115385 baud (0.16% error)
BRAM_COUNT = 30 - 1 # Subtract 1 for the Skid Buffer BRAM on deframer
DSP_COUNT  = 8

# Hand Gesture Task
NET_PATH       = GESTURE_NET_PATH
CLASSES        = GESTURE_CLASSES
NN_CFG         = get_hand_gesture_cfg()

prepare_data   = _prepare_data
get_transforms = _get_transforms