#!/usr/bin/env python3
# nn.globals.py

from pathlib import Path

from nn.constants import (BRAM_CAP, DSP_CAP, LC_CAP, LC_HEADROOM,
                           BUS_WIDTH, BAUD, CLK_FREQ_HZ)
from nn.tasks.hand_gesture.hand_gesture import get_hand_gesture_cfg, GESTURE_CLASSES, GESTURE_NET_PATH
from nn.tasks.hand_gesture.preprocess   import prepare_data as _prepare_data, get_transforms as _get_transforms
from nn.tasks.mnist.mnist      import get_nn_cfg as _get_mnist_cfg, MNIST_NET_PATH, MNIST_CLASSES
from nn.tasks.mnist.preprocess import prepare_mnist_data as _prepare_mnist_data, get_transforms as _get_mnist_transforms
from nn.tasks.generate.generate import GENERATE_NET_PATHS


# Default paths matching the training pipeline
DATAPATH = Path("nn") / "data"
ROMPATH  = DATAPATH / "roms" / "hex"

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
    CLASSES        = MNIST_CLASSES
    prepare_data   = _prepare_mnist_data
    get_transforms = _get_mnist_transforms

    # To load a generated network from the sweep, set SWEEP_IDX to its index (e.g., 30)
    # To load a generated network from the sweep, set SWEEP_IDX to its index (e.g., 30)
    # To use the default MNIST architecture, set SWEEP_IDX to None
    SWEEP_IDX = 308

    if SWEEP_IDX is not None:
        from nn.tasks.generate.generate import generate_networks
        configs = generate_networks()
        NN_CFG = configs[SWEEP_IDX][1]
        NET_PATH = Path("nn") / "tasks" / "generate" / "checkpoints" / f"network_{SWEEP_IDX:04d}.pth"
    else:
        NN_CFG   = _get_mnist_cfg()
        NET_PATH = MNIST_NET_PATH