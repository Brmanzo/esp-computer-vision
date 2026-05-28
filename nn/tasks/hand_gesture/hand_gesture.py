#!/usr/bin/env python3
# nn.tasks.hand_gesture.hand_gesture.py
# Bradley Manzo 2026

from nn.config import NNConfig, InputDimensions, QSchedule
from pathlib import Path

GESTURE_CLASSES = ["okay", "paper", "rock", "call_me"]  # , "rock", "scissors", "fingers_crossed", "rock_on", "thumbs", "up"
GESTURE_NET_PATH = Path("nn") / "tasks" / "hand_gesture" / "gesture_net_quantized.pth"

def get_hand_gesture_cfg(num_classes: int = 4, img_h: int = 240, img_w: int = 320) -> NNConfig:
    '''Returns the standard hardware-aware NNConfig for the hand-gesture recognition task.'''
    return NNConfig(
        input_dimensions = InputDimensions(img_w, img_h),
        in_channels      = [1, 4, 9, 12], # Input Channels per layer
        in_bits          = [1, 3, 3, 4], # Input Bits per layer
        kernels          = [[3,2], [3,2], [3], [1]], # [conv_kernel, pool_kernel]
        padding          = 1, # int or list
        stride           = 1, # int or list
        num_classes      = num_classes, # From dataset
        bias_bits        = [8, 16, 16, 32], # Layer 1 increased to 16-bit: saturating at 8-bit
        q_schedule = [QSchedule( 60, [5, 5, 5, 5, 10, 20], 8, 3),
                      QSchedule( 70, [5, 5, 5,  5, 30], 12, 8),   # 10→9→8, more time at 8
                      QSchedule( 90, [5, 5, 5,  5, 30], 12, 8),   # same
                      QSchedule(100, [5, 5, 5,  5, 50], 12, 8)],    # classifier unchanged

        use_dsp          = [0, 3, 4, 1])


