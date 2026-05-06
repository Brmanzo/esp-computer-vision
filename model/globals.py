from pathlib import Path

from model.config import ModelConfig, InputDimensions
from model.quantize import QSchedule

# Default paths matching the training pipeline
DATAPATH = Path("model") / "data"

BRAM_COUNT = 30 - 1 # Subtract 1 for the Skid Buffer BRAM on deframer
DSP_COUNT  = 8

def get_hand_gesture_cfg(num_classes: int = 8, img_h: int = 240, img_w: int = 320) -> ModelConfig:
    '''Returns the standard hardware-aware ModelConfig for the hand-gesture recognition task.'''
    return ModelConfig(
        input_dimensions = InputDimensions(img_w, img_h),
        in_channels      = [1, 4, 5, 7], # Input Channels per layer
        in_bits          = [1, 1, 1, 5], # Input Bits per layer
        kernels          = [[3,2], [3,2], [3,2], [1]], # [conv_kernel, pool_kernel]
        padding          = 1, # int or list
        stride           = 1, # int or list
        num_classes      = num_classes, # From dataset
        bus_width        = 8, # Decision bit-width
        bias_bits        = [8, 8, 8, 32], # Features use 8-bit, Classifier uses 32-bit to prevent saturation
        q_schedule       = [QSchedule(40, [5, 5, 5, 10, 20], 8, 4),
                            QSchedule(50, [5, 5, 5, 10, 20], 8, 4),
                            QSchedule(60, [5, 5, 5, 10, 20], 8, 4),
                            QSchedule(70, [50], 8, 8)],
        use_dsp          = [0, 0, 1, 2]) # Layer 2 uses 6 DSPs (UseDSP=1), Classifier uses 1 DSP (UseDSP=2)

HAND_GESTURE_CFG = get_hand_gesture_cfg()