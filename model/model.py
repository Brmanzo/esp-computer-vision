# model.py
# Bradley Manzo 2026

import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn

from pathlib import Path

BRAM_COUNT = 30 - 1 # Subtract 1 for the Skid Buffer BRAM on deframer
UART_BUS_WIDTH = 8

from .config import ModelConfig
from .render import render_classifier_layer, render_header, render_wires, render_conv_layer, render_pool_layer, render_footer
from .quantize import QuantConv2d, QuantizeActivation

class cnn_model(nn.Module):
    def render_verilog(self, filepath: Path) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            print(render_header(UART_BUS_WIDTH), file=f)
            # You might need to adjust render_wires to accept self.config.layers
            print(render_wires(self.config), file=f) 
            print("", file=f)

            # Render Feature Layers
            for layer_cfg in self.config.layers:
                print(render_conv_layer(layer_cfg.ConvLayer), file=f)
                if layer_cfg.PoolLayer is not None:
                    print(render_pool_layer(layer_cfg.PoolLayer), file=f)

            # Render Classifier
            print(render_classifier_layer(self.config.classifier_config), file=f)
            
            print(render_footer(), file=f)

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        repo_root = Path(__file__).resolve().parents[1]
        out_file = repo_root / "rtl" / "blocks" / "cnn.sv"

        # 1. Dynamically Build the Feature Extractor
        feature_layers = []
        
        # Iterate naturally over all feature layers
        for layer_cfg in self.config.layers:
            conv = layer_cfg.ConvLayer
            pool = layer_cfg.PoolLayer
            
            feature_layers.append(QuantConv2d(conv._in_ch, conv._out_ch, kernel_size=conv._kernel_width, padding=conv._padding, bias=False))
            out_ch = conv._out_ch
            assert out_ch is not None
            feature_layers.append(nn.BatchNorm2d(out_ch))
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
            # Notice the 1x1 kernel assumption typical for fully convolutional classifiers
            QuantConv2d(cls_cfg._in_ch, cls_cfg._num_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(cls_cfg._num_classes)
        )

        # 3. Utilities
        self.ram_utilization()
        self.render_verilog(out_file)

        # 3. Utilities
        self.ram_utilization()
        self.render_verilog(out_file)

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
                
            # Access the pre-calculated width directly from the config object!
            if conv._input_dims.width is None:
                raise ValueError(f"Input dimensions for layer {conv._layer_num} are not defined. Cannot calculate RAM utilization.")
            target_ram_bits = 16 if (conv._input_dims.width - 1) <= 256 else 8
            channels_per_ram = target_ram_bits // (conv._kernel_width - 1) * conv._in_bits
            rams_per_layer = (conv._in_ch + channels_per_ram - 1) // channels_per_ram
            
            rams -= rams_per_layer
        
        assert rams >= 0, f"Model exceeds BRAM budget! Remaining: {rams}"
        print(f"BRAMs remaining after model layers: {rams}")