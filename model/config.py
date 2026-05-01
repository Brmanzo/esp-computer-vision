# model.config.py
# Bradley Manzo 2026

import math
from typing import List, Optional

from .quantize import QSchedule

class InputDimensions:
    '''Organizes spatial input dimensions, as well as term_count for the classifier.'''
    def __init__(self, width: Optional[int], height: Optional[int]):
        self.width  = width
        self.height = height
        if width is not None and height is not None:
            self.term_count = width * height

class ConvConfig:
    '''Parameterizes each convolution layer for pytorch training, hardware generation, and cocotb verification.'''
    def __init__(self, in_ch: int, in_bits: int,
                 out_bits: int, kernels: List[List[int]], stride: int, 
                 padding: int, q_schedule: QSchedule, layer_num: int, bias_bits: int, 
                 out_ch: Optional[int] = None, input_dims: Optional[InputDimensions] = None):
        
        self._input_dims = input_dims or InputDimensions(None, None)
        self._layer_num  = layer_num

        self._in_bits  = in_bits
        self._out_bits = out_bits
        self._bias_bits = bias_bits

        # Convolution kernel is always the first element of this layer's kernel width
        self._kernel_width = kernels[layer_num][0]
        self._q_schedule   = q_schedule
        self._weight_bits  = q_schedule._q_min_bits

        self._in_ch  = in_ch
        self._out_ch = out_ch

        self._stride  = stride
        self._padding = padding

        # Input logic
        # If first layer, always take top level inputs
        if self._layer_num == 0:
            self._valid_i = "valid_i"
            self._data_i  = "data_i"
            self._ready_o = "ready_o"
        else: 
            self._ready_o = f"conv_{self._layer_num}_ready"
            # If the last module in the previous layer is a conv, connect to that conv
            if len(kernels[self._layer_num - 1]) == 1:
                self._valid_i = f"conv_{self._layer_num - 1}_valid"
                self._data_i  = f"conv_{self._layer_num - 1}_data"
            # Otherwise, connect to the previous layer's pool
            else:
                self._valid_i = f"pool_{self._layer_num - 1}_valid"
                self._data_i  = f"pool_{self._layer_num - 1}_data"

        # Output logic
        self._valid_o = f"conv_{self._layer_num}_valid"
        self._data_o  = f"conv_{self._layer_num}_data"
        # If this is the last layer, connect to classifier
        if self._layer_num == len(kernels) - 2:
            self._ready_i = f"classifier_ready"
        # If the next layer is a conv, connect to that conv
        elif len(kernels[self._layer_num]) == 1:
            self._ready_i = f"conv_{self._layer_num + 1}_ready"
        # Otherwise, connect to the next layer's pool
        else:
            self._ready_i = f"pool_{self._layer_num}_ready"

class PoolConfig:
    '''Parameterizes each pooling layer for pytorch training, hardware generation, and cocotb verification.'''
    def __init__(self, in_ch: int, in_bits: int, kernel: int, mode: str, layer_num: int, 
                 line_width_px: Optional[int] = None, line_count_px: Optional[int] = None):
        self._input_dims = InputDimensions(line_width_px, line_count_px)
        self._layer_num  = layer_num

        self._in_bits  = in_bits
        self._out_bits = in_bits  # Pooling doesn't change bit-width

        self._kernel_width = kernel

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
    '''Parameterizes the final classifier layer for pytorch training, hardware generation, and cocotb verification.'''
    def __init__(self, in_ch: int, in_bits: int, out_bits: int, num_classes: int, bias_bits: int,
                 q_schedule: QSchedule, layer_num: int, line_width_px: Optional[int] = None, line_count_px: Optional[int] = None):
        self._in_ch       = in_ch
        self._in_bits     = in_bits
        self._out_bits    = out_bits
        self._bias_bits   = bias_bits
        self._num_classes = num_classes
        self._q_schedule  = q_schedule
        self._layer_num   = layer_num

        self._term_count = InputDimensions(line_width_px, line_count_px).term_count  # Classifier doesn't have spatial dimensions

        # Classifier layers are always connected to the final solitary conv layer
        self._valid_i = f"conv_{self._layer_num - 1}_valid"
        self._data_i  = f"conv_{self._layer_num - 1}_data"

        # Classifier layers are always followed by the next layer's conv block
        self._ready_i = f"ready_i"

        # Classifier layers are never at the immediate head or tail, so we don't need to
        # account for global input or output connections here.
        self._ready_o = f"classifier_ready"
        self._valid_o = f"valid_o"
        self._data_o  = f"data_o"

class LayerConfig:
    '''Each layer is comprised of a convolution layer and potentially a pooling layer.'''
    def __init__(self, ConvLayer: ConvConfig, PoolLayer: Optional[PoolConfig] = None):
        self.ConvLayer = ConvLayer
        self.PoolLayer = PoolLayer

class ModelConfig:
    '''Accepts high-level model specs, and translates into detailed layer-by-layer configurations.'''
    def full_precision_acc_bits(self, in_channels, kernel_size, in_bits, weight_bits):
        '''Calculates full precision output width for convolution.'''
        term_count = in_channels * (kernel_size ** 2)
        acc_bits   = in_bits + weight_bits + math.ceil(math.log2(term_count))
        return acc_bits

    def __init__(self, input_dimensions: InputDimensions, in_channels: List[int], 
                 in_bits: List[int], kernels: List[List[int]], 
                 stride:  List[int] | int, padding: List[int] | int, bias_bits: int,
                 num_classes: int, bus_width: int, q_schedule: List[QSchedule]):
        
        # In channels length defines the number of layers
        self.num_layers = len(in_channels)
        # Input dimensions of the first layer are provided and subsequent layers are inferred based on conv/pool parameters
        self.in_dims    = input_dimensions
        self._in_bits   = in_bits
        self._bus_width = bus_width
        self._bias_bits = bias_bits
        
        # Determine out_channels for all layers (including classifier)
        out_channels = in_channels[1:] + [num_classes]

        # The master list holding the structured layer objects
        self.layers: List[LayerConfig] = []

        # Quantization schedule for each layer, used in training, and also to determine bit-widths for weights and activations
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
            
            # Stride and padding can be fixed for every layer, or layer-specific
            c_stride = stride[i]  if isinstance(stride, list) else stride
            c_pad    = padding[i] if isinstance(padding, list) else padding
            # Weight bits are determined by the final quantized width
            c_weight_bits = q_schedule[i]._q_min_bits
            
            # 1. Determine In Bits
            if i < len(in_bits):
                c_in_bits = in_bits[i]
            else:
                c_in_bits = current_out_bits
                assert c_in_bits is not None, "Input bit-width is undefined for this layer"

            # 2. Check if this is the final layer (Classifier)
            if i == self.num_layers - 1:
                self.classifier_config = ClassifierConfig(
                    in_ch=c_in_ch, 
                    in_bits=c_in_bits, 
                    out_bits=self._bus_width, # Classifier output maps to the UART Bus Width
                    num_classes=num_classes, 
                    q_schedule=q_schedule[i], 
                    layer_num=i,
                    bias_bits=self._bias_bits,
                    line_width_px=current_w, 
                    line_count_px=current_h
                )
                break # Classifier always concludes the model construction

            # 3. Otherwise, determine Out Bits for standard Conv layer
            if i < len(in_bits) - 1:
                c_out_bits = in_bits[i+1]
            else:
                c_out_bits = self.full_precision_acc_bits(c_in_ch, c_kernel, c_in_bits, c_weight_bits)

            # 4. Create ConvConfig
            conv_cfg = ConvConfig(
                in_ch=c_in_ch, in_bits=c_in_bits, out_bits=c_out_bits,
                kernels=kernels, stride=c_stride, padding=c_pad, bias_bits=self._bias_bits,
                q_schedule=q_schedule[i], out_ch=c_out_ch, layer_num=i,
                input_dims=InputDimensions(current_w, current_h)
            )
    
            # 5. Update Spatial Dimensions (Post-Conv)
            # Next layer's input dimensions are reduced by kernel width - 1 if not padding
            if current_w is not None:
                current_w = ((current_w + 2 * c_pad - c_kernel) // c_stride) + 1
            if current_h is not None:
                current_h = ((current_h + 2 * c_pad - c_kernel) // c_stride) + 1

            # 6. Check for Pooling & Update Spatial Dimensions (Post-Pool)
            pool_cfg = None
            # If current layer kernel list specifies a second kernel width, then we construct a pooling layer
            if len(kernels[i]) > 1:
                p_kernel = kernels[i][1]
                pool_cfg = PoolConfig(
                    in_ch=c_out_ch, in_bits=c_out_bits, kernel=p_kernel, mode="max",
                    layer_num=i, line_width_px=current_w, line_count_px=current_h
                )
                # Pooling divides next layer's input dimensions by kernel width
                if current_w is not None:
                    current_w = ((current_w - p_kernel) // p_kernel) + 1
                if current_h is not None:
                    current_h = ((current_h - p_kernel) // p_kernel) + 1

            # 7. Store Layer & Update State
            self.layers.append(LayerConfig(ConvLayer=conv_cfg, PoolLayer=pool_cfg))
            current_out_bits = c_out_bits