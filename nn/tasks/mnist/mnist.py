#!/usr/bin/env python3
# nn.tasks.mnist.mnist.py

from pathlib import Path
from nn.config import NNConfig, InputDimensions
from nn.quantize import QSchedule


def get_nn_cfg(num_classes: int = 10, img_h: int = 28, img_w: int = 28) -> NNConfig:
    '''Returns the hardware-aware NNConfig for MNIST digit classification.'''
    return NNConfig(
        input_dimensions = InputDimensions(img_w, img_h),
        in_channels      = [1, 4, 9, 12],
        in_bits          = [1, 2, 3, 3],
        kernels          = [[3,2], [3,2], [3], [1]],
        padding          = 1,
        stride           = 1,
        num_classes      = num_classes,
        bias_bits        = [8, 16, 16, 16],
        q_schedule = [QSchedule(25, [3, 3, 3, 3, 5, 10], 8, 3),   # L0:  25 warmup, drops 8→3
                      QSchedule(30, [3, 3, 3, 3, 15],  12, 8),  # L1:  30 warmup, drops 12→8
                      QSchedule(40, [3, 3, 3, 3, 20],  12, 8),  # L2:  40 warmup, drops 12→8
                      QSchedule(50, [3, 3, 3, 3, 25],  12, 8)],  # Cls: 50 warmup, drops 12→8
        use_dsp    = [0, 3, 4, 1])

MNIST_NET_PATH = Path("nn") / "tasks" / "mnist" / "mnist_net_quantized.pth"
MNIST_CLASSES = [str(d) for d in range(10)]
NN_CFG = get_nn_cfg()