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
        in_channels      = [1, 4, 8, 12], # Input Channels per layer
        in_bits          = [1, 2, 2, 4], # Input Bits per layer
        kernels          = [[3,2], [3,2], [3,2], [1]], # [conv_kernel, pool_kernel]
        padding          = 1, # int or list
        stride           = 1, # int or list
        num_classes      = num_classes, # From dataset
        bus_width        = 8, # Decision bit-width
        bias_bits        = [8, 8, 16, 32], # Increased Layer 2 to 16-bit to prevent saturation
        q_schedule       = [QSchedule( 60, [5, 5, 5, 5, 5, 10, 20], 8, 2),
                            QSchedule( 70, [5],  8, 8),
                            QSchedule( 90, [5],  8, 8),
                            QSchedule(100, [50], 8, 8)],
        use_dsp          = [0, 2, 4, 2])

HAND_GESTURE_CFG = get_hand_gesture_cfg()