# model.model.py
# Bradley Manzo 2026

import torch
import torch.nn as nn

BRAM_COUNT = 30 - 1 # Subtract 1 for the Skid Buffer BRAM on deframer

from model.config   import ModelConfig
from model.quantize import QuantConv2d, QuantizeActivation

class cnn_model(nn.Module):
    '''Construct the Pytorch CNN model based on provided Model Config.'''
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 1. Dynamically Build the Feature Extractor
        feature_layers: list[nn.Module] = []
        
        # Iterate naturally over all feature layers in model config
        for layer_cfg in self.config.layers:
            conv = layer_cfg.ConvLayer
            pool = layer_cfg.PoolLayer
            
            feature_layers.append(QuantConv2d(
                in_channels=conv._in_ch, 
                out_channels=conv._out_ch, 
                kernel_size=conv._kernel_width, 
                padding=conv._padding, 
                weight_bits=conv._weight_bits,
                bias_bits=conv._bias_bits,
                bias=False
            ))
            
            # Stabilization: Add ReLU before quantization
            feature_layers.append(nn.ReLU(inplace=True))
            feature_layers.append(QuantizeActivation(bits=conv._out_bits))
            
            if pool is not None:
                if pool._mode == 0:
                    feature_layers.append(nn.MaxPool2d(pool._kernel_width))
                else:
                    feature_layers.append(nn.AvgPool2d(pool._kernel_width))

        self.features = nn.Sequential(*feature_layers)

        # 2. Build the Classifier Block 
        cls_cfg = config.classifier_config
        
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.1),
            # 1x1 kernel for fully convolutional classifiers
            QuantConv2d(
                in_channels=cls_cfg._in_ch, 
                out_channels=cls_cfg._num_classes, 
                kernel_size=1, 
                weight_bits=cls_cfg._q_schedule._q_min_bits,
                bias_bits=cls_cfg._bias_bits,
                bias=False
            ),
        )
        
        # 3. Utilities
        self.ram_utilization()
        # Update cnn.sv with current architecture
        from model.render import render_verilog
        render_verilog(self.config)

    def forward(self, x):
        x = self.features(x)           
        
        # Global Max
        # Reduces (Batch, Channels, H, W) -> (Batch, Channels, 1, 1)
        x = torch.amax(x, dim=(2, 3), keepdim=True) 
        
        # Classifier
        x = self.classifier(x)         
        
        # Flatten the final output to (Batch, Num_Classes)
        x = torch.flatten(x, 1)
        
        return x

    def ram_utilization(self) -> None:
        rams = BRAM_COUNT
        
        for layer_cfg in self.config.layers:
            conv = layer_cfg.ConvLayer
            if conv._kernel_width == 1:
                continue # Skip 1x1 convolutions (classifier)
                
            # Access the pre-calculated width directly from config
            if conv._input_dims.width is None:
                raise ValueError(f"Input dimensions for layer {conv._layer_num} are not defined. Cannot calculate RAM utilization.")
            target_ram_bits = 16 if (conv._input_dims.width - 1) <= 256 else 8
            channels_per_ram = target_ram_bits // (conv._kernel_width - 1) * conv._in_bits
            rams_per_layer = (conv._in_ch + channels_per_ram - 1) // channels_per_ram
            
            rams -= rams_per_layer
        
        assert rams >= 0, f"Model exceeds BRAM budget! Remaining: {rams}"
        print(f"BRAMs remaining after model layers: {rams}")