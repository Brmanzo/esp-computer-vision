#!/usr/bin/env python3
# nn.globals.py

from pathlib import Path

from nn.config import NNConfig, InputDimensions
from nn.quantize import QSchedule

# Default paths matching the training pipeline
DATAPATH = Path("nn") / "data"
ROMPATH = DATAPATH / "roms" / "hex"

BAUD = 115200  # 12 MHz / (prescale=13 * 8) ≈ 115385 baud (0.16% error)
BRAM_COUNT = 30 - 1 # Subtract 1 for the Skid Buffer BRAM on deframer
DSP_COUNT  = 8

MNIST_CLASSES = [str(d) for d in range(10)]

def get_mnist_cfg(num_classes: int = 10, img_h: int = 28, img_w: int = 28) -> NNConfig:
    '''Returns the hardware-aware NNConfig for MNIST digit classification.'''
    return NNConfig(
        input_dimensions = InputDimensions(img_w, img_h),
        in_channels      = [1, 4, 9, 12],
        in_bits          = [1, 3, 3, 4],
        kernels          = [[3,2], [3,2], [3], [1]],
        padding          = 1,
        stride           = 1,
        num_classes      = num_classes,
        bus_width        = 8,
        bias_bits        = [8, 16, 16, 32],
        q_schedule = [QSchedule(25, [3, 3, 3, 3, 5, 10], 8, 3),   # L0:  25 warmup, drops 8→3
                      QSchedule(30, [3, 3, 3, 3, 15],   8, 4),  # L1:  30 warmup, drops 12→8
                      QSchedule(40, [3, 3, 3, 3, 20],   8, 4),  # L2:  40 warmup, drops 12→8
                      QSchedule(50, [3, 3, 3, 3, 25],   8, 4)],  # Cls: 50 warmup, drops 12→8
        use_dsp          = [0, 3, 4, 1])

MNIST_CFG = get_mnist_cfg()