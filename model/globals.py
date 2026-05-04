from pathlib import Path

from model.config import ModelConfig, InputDimensions
from model.quantize import QSchedule

# Default paths matching the training pipeline
DATAPATH = Path("model") / "data"

def get_hand_gesture_cfg(num_classes: int = 10, img_h: int = 240, img_w: int = 320) -> ModelConfig:
    '''Returns the standard hardware-aware ModelConfig for the hand-gesture recognition task.'''
    return ModelConfig(
        input_dimensions = InputDimensions(img_w, img_h),
        in_channels      = [1, 8, 16, 24, 32],
        in_bits          = [1, 1, 1, 1, -1], # -1 indicates full precision
        kernels          = [[3,2], [3,2], [3,2], [3], [1]],
        padding          = 1,
        stride           = 1,
        num_classes      = num_classes,
        bus_width        = 8,
        bias_bits        = [8, 8, 8, 8, 32], # Features use 8-bit, Classifier uses 32-bit to prevent saturation
        q_schedule       = [QSchedule(20, [5, 5, 5, 10, 20], 8, 4),
                            QSchedule(30, [5, 5, 5, 10, 20], 8, 4),
                            QSchedule(40, [5, 5, 5, 10, 20], 8, 4),
                            QSchedule(50, [5, 5, 5, 10, 20], 8, 4),
                            QSchedule(50, [10], 13, 13)])

HAND_GESTURE_CFG = get_hand_gesture_cfg()