import math
from typing import List, Optional

from .quantize import QSchedule

class InputDimensions:
    def __init__(self, width: Optional[int], height: Optional[int]):
        self.width  = width
        self.height = height
        if width is not None and height is not None:
            self.term_count = width * height

class ConvConfig:
    def __init__(self, in_ch: int, in_bits: int,
                 out_bits: int, kernels: List[List[int]], stride: int, 
                 padding: int, q_schedule: QSchedule, layer_num: int, 
                 out_ch: Optional[int] = None, input_dims: Optional[InputDimensions] = None):
        
        self._input_dims = input_dims or InputDimensions(None, None)
        self._layer_num = layer_num

        self._in_bits = in_bits
        self._out_bits = out_bits

        self._kernel_width = kernels[layer_num][0]
        self._q_schedule   = q_schedule
        self._weight_bits  = q_schedule._q_min_bits

        self._in_ch  = in_ch
        self._out_ch = out_ch

        self._stride = stride
        self._padding = padding

        # Input logic
        if self._layer_num == 0:
            self._valid_i = "valid_i"
            self._data_i  = "data_i"
            self._ready_o = "ready_o"
        else: 
            self._ready_o = f"conv_{self._layer_num}_ready"
            if len(kernels[self._layer_num - 1]) == 1:  # If the last module in the previous layer is a conv, connect to that conv
                self._valid_i = f"conv_{self._layer_num - 1}_valid"
                self._data_i  = f"conv_{self._layer_num - 1}_data"
                # Otherwise, connect to the previous layer's pool
            else:
                self._valid_i = f"pool_{self._layer_num - 1}_valid"
                self._data_i  = f"pool_{self._layer_num - 1}_data"

        # Output logic
        self._valid_o = f"conv_{self._layer_num}_valid"
        self._data_o  = f"conv_{self._layer_num}_data"
        if self._layer_num == len(kernels) - 2:  # If this is the last layer, connect to classifier
            self._ready_i = f"classifier_ready"
        elif len(kernels[self._layer_num]) == 1:
            self._ready_i = f"conv_{self._layer_num + 1}_ready"
        else:
            self._ready_i = f"pool_{self._layer_num}_ready"

class PoolConfig:
    def __init__(self, in_ch: int, in_bits: int, kernel: int, mode: str, layer_num: int, 
                 line_width_px: Optional[int] = None, line_count_px: Optional[int] = None):
        self._input_dims = InputDimensions(line_width_px, line_count_px)
        self._layer_num = layer_num

        self._in_bits  = in_bits
        self._out_bits = in_bits  # Pooling doesn't change bit-width

        self._kernel_width   = kernel

        self._in_ch    = in_ch
        self._out_ch   = in_ch  # Pooling doesn't change channel count

        self._mode     = 0 if mode == "max" else 1

        # Pool layers are always connected to the conv layer in the same block
        self._valid_i = f"conv_{self._layer_num}_valid"
        self._data_i  = f"conv_{self._layer_num}_data"

        # Pool layers are always followed by the next layer's conv block
        self._ready_i = f"conv_{self._layer_num + 1}_ready"

        # Pool layers are never at the immediate head or tail, so we don't need to
        # account for global input or output connections here.
        self._ready_o = f"pool_{self._layer_num}_ready"
        self._valid_o = f"pool_{self._layer_num}_valid"
        self._data_o  = f"pool_{self._layer_num}_data"

class ClassifierConfig:
    def __init__(self, in_ch: int, in_bits: int, out_bits: int, num_classes: int, 
                 q_schedule: QSchedule, layer_num: int, line_width_px: Optional[int] = None, line_count_px: Optional[int] = None):
        self._in_ch = in_ch
        self._in_bits = in_bits
        self._out_bits = out_bits
        self._num_classes = num_classes
        self._q_schedule = q_schedule
        self._layer_num = layer_num

        self._term_count = InputDimensions(line_width_px, line_count_px).term_count  # Classifier doesn't have spatial dimensions

        # Pool layers are always connected to the final solitary conv layer
        self._valid_i = f"conv_{self._layer_num - 1}_valid"
        self._data_i  = f"conv_{self._layer_num - 1}_data"

        # Pool layers are always followed by the next layer's conv block
        self._ready_i = f"ready_i"

        # Pool layers are never at the immediate head or tail, so we don't need to
        # account for global input or output connections here.
        self._ready_o = f"classifier_ready"
        self._valid_o = f"valid_o"
        self._data_o  = f"data_o"

class LayerConfig:
    def __init__(self, ConvLayer: ConvConfig, PoolLayer: Optional[PoolConfig] = None):
        self.ConvLayer = ConvLayer
        self.PoolLayer = PoolLayer

class ModelConfig:
    def full_precision_acc_bits(self, in_channels, kernel_size, in_bits, weight_bits):
        term_count = in_channels * (kernel_size ** 2)
        acc_bits = in_bits + weight_bits + math.ceil(math.log2(term_count))
        return acc_bits

    def __init__(self, input_dimensions: InputDimensions, in_channels: List[int], 
                 in_bits: List[int], kernels: List[List[int]], 
                 stride: List[int] | int, padding: List[int] | int,
                 num_classes: int, bus_width: int, q_schedule: List[QSchedule]):
        
        self.num_layers = len(in_channels)
        self.in_dims    = input_dimensions
        self._in_bits   = in_bits
        
        # Determine out_channels for all layers (including classifier)
        out_channels = in_channels[1:] + [num_classes]

        # The master list holding our structured layer objects
        self.layers: List[LayerConfig] = []
        self.q_schedule = q_schedule

        # --- Running State Variables ---
        # These track the dimensions and bit-widths as data flows through the network
        current_w = input_dimensions.width
        current_h = input_dimensions.height
        current_out_bits = None # Will be set at the end of each loop

        for i in range(self.num_layers):
            c_in_ch  = in_channels[i]
            c_out_ch = out_channels[i]
            c_kernel = kernels[i][0]
            
            c_stride = stride[i]  if isinstance(stride, list) else stride
            c_pad    = padding[i] if isinstance(padding, list) else padding
            c_weight_bits = q_schedule[i]._q_min_bits
            
            # 1. Determine In Bits
            if i < len(in_bits):
                c_in_bits = in_bits[i]
            else:
                c_in_bits = current_out_bits
                assert c_in_bits is not None, "Input bit-width is undefined for this layer"

            # 2. Check if this is the FINAL layer (Classifier)
            if i == self.num_layers - 1:
                self.classifier_config = ClassifierConfig(
                    in_ch=c_in_ch, 
                    in_bits=c_in_bits, 
                    out_bits=bus_width, # Classifier output maps to the UART Bus Width
                    num_classes=num_classes, 
                    q_schedule=q_schedule[i], 
                    layer_num=i,
                    line_width_px=current_w, 
                    line_count_px=current_h
                )
                break # We are done, classifier is always the end of the line!

            # 3. Otherwise, determine Out Bits for standard Conv layer
            if i < len(in_bits) - 1:
                c_out_bits = in_bits[i+1]
            else:
                c_out_bits = self.full_precision_acc_bits(c_in_ch, c_kernel, c_in_bits, c_weight_bits)

            # 4. Create ConvConfig
            conv_cfg = ConvConfig(
                in_ch=c_in_ch, in_bits=c_in_bits, out_bits=c_out_bits,
                kernels=kernels, stride=c_stride, padding=c_pad, 
                q_schedule=q_schedule[i], out_ch=c_out_ch, layer_num=i,
                input_dims=InputDimensions(current_w, current_h)
            )
    
            # 5. Update Spatial Dimensions (Post-Conv)
            if current_w is not None:
                current_w = ((current_w + 2 * c_pad - c_kernel) // c_stride) + 1
            if current_h is not None:
                current_h = ((current_h + 2 * c_pad - c_kernel) // c_stride) + 1

            # 6. Check for Pooling & Update Spatial Dimensions (Post-Pool)
            pool_cfg = None
            if len(kernels[i]) > 1:
                p_kernel = kernels[i][1]
                pool_cfg = PoolConfig(
                    in_ch=c_out_ch, in_bits=c_out_bits, kernel=p_kernel, mode="max",
                    layer_num=i, line_width_px=current_w, line_count_px=current_h
                )
                if current_w is not None:
                    current_w = ((current_w - p_kernel) // p_kernel) + 1
                if current_h is not None:
                    current_h = ((current_h - p_kernel) // p_kernel) + 1

            # 7. Store Layer & Update State
            self.layers.append(LayerConfig(ConvLayer=conv_cfg, PoolLayer=pool_cfg))
            current_out_bits = c_out_bits