#!/usr/bin/env python3
# nn.globals.py

from pathlib import Path

from nn.config import InputDimensions
from nn.tasks.hand_gesture.hand_gesture import get_hand_gesture_cfg, GESTURE_CLASSES, GESTURE_NET_PATH
from nn.tasks.hand_gesture.preprocess   import prepare_data as _prepare_data, get_transforms as _get_transforms
from nn.tasks.mnist.mnist      import get_nn_cfg as _get_mnist_cfg, MNIST_NET_PATH, MNIST_CLASSES
from nn.tasks.mnist.preprocess import prepare_mnist_data as _prepare_mnist_data, get_transforms as _get_mnist_transforms

# Default paths matching the training pipeline
DATAPATH = Path("nn") / "data"
ROMPATH  = DATAPATH / "roms" / "hex"

BRAM_COUNT  = 30 - 1 # Subtract 1 for the Skid Buffer BRAM on deframer
DSP_COUNT   = 8
BAUD        = 115200  # 12 MHz / (prescale=13 * 8) ≈ 115385 baud (0.16% error)
CLK_FREQ_HZ = 12_000_000  # 12 MHz Clock
BUS_WIDTH   = 8

CURRENT_TASK = "mnist"  # "mnist" or "hand_gesture"

if CURRENT_TASK == "hand_gesture":
    # Hand Gesture Task
    NET_PATH       = GESTURE_NET_PATH
    CLASSES        = GESTURE_CLASSES
    NN_CFG         = get_hand_gesture_cfg()
    prepare_data   = _prepare_data
    get_transforms = _get_transforms
elif CURRENT_TASK == "mnist":
    # MNIST Task
    NET_PATH       = MNIST_NET_PATH
    CLASSES        = MNIST_CLASSES
    NN_CFG         = _get_mnist_cfg()
    prepare_data   = _prepare_mnist_data
    get_transforms = _get_mnist_transforms