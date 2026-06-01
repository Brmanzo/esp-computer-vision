#!/usr/bin/env python3
# nn.globals.py

from pathlib import Path

from nn.constants import (BRAM_CAP, DSP_CAP, LC_CAP, LC_HEADROOM,
                           BUS_WIDTH, BAUD, CLK_FREQ_HZ)
from nn.tasks.hand_gesture.hand_gesture import get_hand_gesture_cfg, GESTURE_CLASSES, GESTURE_NET_PATH
from nn.tasks.hand_gesture.preprocess   import prepare_data as _prepare_data, get_transforms as _get_transforms
from nn.tasks.mnist.mnist      import get_nn_cfg as _get_mnist_cfg, MNIST_NET_PATH, MNIST_CLASSES
from nn.tasks.mnist.preprocess import prepare_mnist_data as _prepare_mnist_data, get_transforms as _get_mnist_transforms
from nn.sweep.generate import GENERATE_NET_PATHS

# Default paths matching the training pipeline
DATAPATH = Path("nn") / "data"
ROMPATH  = DATAPATH / "roms" / "hex"

CURRENT_TASK = "mnist"  # "mnist" or "hand_gesture"

# To load a generated network from the sweep, set SWEEP_IDX to its index (e.g., 308)
# To use the default custom architecture for the CURRENT_TASK, set SWEEP_IDX to None
SWEEP_IDX = 308

if CURRENT_TASK == "hand_gesture":
    # Hand Gesture Task
    CLASSES        = GESTURE_CLASSES
    prepare_data   = _prepare_data
    get_transforms = _get_transforms
    _BASE_CFG      = get_hand_gesture_cfg()
    _DEFAULT_PATH  = GESTURE_NET_PATH
elif CURRENT_TASK == "mnist":
    # MNIST Task
    CLASSES        = MNIST_CLASSES
    prepare_data   = _prepare_mnist_data
    get_transforms = _get_mnist_transforms
    _BASE_CFG      = _get_mnist_cfg()
    _DEFAULT_PATH  = MNIST_NET_PATH

if SWEEP_IDX is not None:
    from nn.sweep.generate import generate_networks
    configs = generate_networks(_BASE_CFG)
    NN_CFG = configs[SWEEP_IDX][1]
    NET_PATH = GENERATE_NET_PATHS / f"network_{SWEEP_IDX:04d}.pth"
else:
    NN_CFG   = _BASE_CFG
    NET_PATH = _DEFAULT_PATH