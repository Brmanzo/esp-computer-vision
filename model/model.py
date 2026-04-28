# model.py
# Bradley Manzo 2026

import torch
import torch.nn as nn
import math
from typing import cast

import torch
import torch.nn as nn

from pathlib import Path

BRAM_COUNT = 30 - 1 # Subtract 1 for the Skid Buffer BRAM on deframer
UART_BUS_WIDTH = 8

from .render import render_classifier_layer, render_header, render_wires, render_conv_layer, render_pool_layer, render_linear_layer, render_footer
from .quantize import QuantConv2d, QuantizeActivation

class cnn_model(nn.Module):

    def ram_utilization(self) -> None:
        rams = BRAM_COUNT
        # Each BRAM can support 4 input channels
        for layer in range(len(self._kernels)):
            for module in range(len(self._kernels[layer])):
                if (self._kernels[layer][module] - 1) <= 0:
                    continue # Skip layers with kernel size 1 (e.g., classifier)
                target_ram_bits  = 16 if (self._input_widths[layer][module] - 1) <= 256 else 8
                channels_per_ram = target_ram_bits // (self._kernels[layer][module] - 1) * self._input_bits[layer]
                rams_per_layer = (self._in_ch[layer] + channels_per_ram - 1) // channels_per_ram
                
                rams -= rams_per_layer
        
        assert rams >= 0, f"Model exceeds BRAM budget! Remaining: {rams}"
        print(f"BRAMs remaining after model layers: {rams}")

    def render_verilog(self, filepath: Path) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            print(render_header(UART_BUS_WIDTH), file=f)
            print(render_wires(self._kernels), file=f)
            print("", file=f)

            pool_modes = []
            for m in self.features.children():
                if isinstance(m, nn.MaxPool2d):
                    pool_modes.append(0)
                elif isinstance(m, nn.AvgPool2d):
                    pool_modes.append(1)

            for layer in range(self._layers):
                for module in range(len(self._kernels[layer])):
                    if module == 0:  # Convolution module
                        print(
                            render_conv_layer(
                                LineWidthPx=self._input_widths[layer][0],
                                LineCountPx=self._input_heights[layer][0],
                                InBits=self._input_bits[layer],
                                OutBits=self._out_bits[layer],
                                KernelWidth=self._kernels[layer][0],
                                WeightBits=self._weight_bits[layer],
                                InChannels=self._in_ch[layer],
                                OutChannels=self._out_ch[layer],
                                Stride=self._stride[layer],
                                Weights=f"weights_{layer}",
                                instance=layer,
                                num_instances=self._layers,
                                kernels=self._kernels,
                                padding=self._padding[layer],
                            ),
                            file=f,
                        )
                    elif module == 1:  # Pooling module
                        print(
                            render_pool_layer(
                                LineWidthPx=self._input_widths[layer][1],
                                LineCountPx=self._input_heights[layer][1],
                                InBits=self._input_bits[layer],
                                KernelWidth=self._kernels[layer][1],
                                InChannels=self._out_ch[layer],
                                instance=layer,
                                num_instances=self._layers,
                                kernels=self._kernels,
                                PoolMode=pool_modes[layer], # 0 for max pooling, 1 for average pooling
                            ),
                            file=f,
                        )
            print(
                render_classifier_layer(
                    TermBits=self._out_bits[-1], 
                    TermCount=self._classifier_term_count,
                    BusBits=UART_BUS_WIDTH,
                    InChannels=self._in_ch[-1],
                    ClassCount=self._num_classes,
                    WeightBits=self._weight_bits[-1],
                    BiasBits=8, # Assuming 8 bits for biases, can be adjusted as needed
                    Weights=f"classifier_weights",
                    Biases=f"classifier_biases",
                    instance=self._layers
                ),
                file=f
            )
            print(render_footer(), file=f)

    def __init__(self, input_dimensions, in_channels, in_bits, kernels, padding, schedule, num_classes=5):
        super().__init__()

        repo_root = Path(__file__).resolve().parents[1]   # adjust depth as needed
        out_file = repo_root / "rtl" / "blocks" / "cnn.sv"

        self._in_ch       = in_channels
        self._out_ch      = in_channels[1:] + [num_classes]
        self._num_classes = num_classes
        self._layers = len(self._in_ch)
        
        self._input_bits       = in_bits
        self._out_bits         = in_bits[1:]
        self._padding          = padding
        self._kernels          = kernels

        self._input_dimensions = input_dimensions
        w, h = self._input_dimensions

        self._input_widths     = [[] for _ in range(self._layers)]
        self._input_heights    = [[] for _ in range(self._layers)]
        
        for layer in range(self._layers):
            for module in range(len(self._kernels[layer])):
                self._input_widths[layer].append(w)
                self._input_heights[layer].append(h)

                if module == 0: # Convolution module
                    w = w - self._kernels[layer][module] + 1 + 2 * self._padding[layer]
                    h = h - self._kernels[layer][module] + 1 + 2 * self._padding[layer]

                elif module == 1: # Pooling module
                    w = w // self._kernels[layer][1]
                    h = h // self._kernels[layer][1]

        self._classifier_term_count = self._input_widths[-1][0] * self._input_heights[-1][0]

        self._weight_bits   = [schedule[i]._q_min_bits for i in range(self._layers)]

        self._stride        = [    1  for _ in range(self._layers)]

        full_precision_term_count = self._in_ch[3] * (self._kernels[3][0] ** 2)
        full_precision_acc_bits = self._out_bits[2] + self._weight_bits[3] + math.ceil(math.log2(full_precision_term_count))

        classifier_term_count = self._in_ch[4] * (self._kernels[4][0] ** 2)
        classifier_acc_bits = full_precision_acc_bits + self._weight_bits[4] + math.ceil(math.log2(classifier_term_count))

        self._input_bits = in_bits + [full_precision_acc_bits]
        self._out_bits = in_bits[1:] + [full_precision_acc_bits, classifier_acc_bits]
        
        # One BRAM consumed by binary in_ch
        self.ram_utilization()

        self.features = nn.Sequential(
            # Block 0
            QuantConv2d(self._in_ch[0], self._out_ch[0], kernel_size=self._kernels[0][0], padding=self._padding[0], bias=False),
            nn.BatchNorm2d(self._out_ch[0]),
            QuantizeActivation(bits=self._out_bits[0]),
            nn.MaxPool2d(self._kernels[0][1]),

            # Block 1
            QuantConv2d(self._in_ch[1], self._out_ch[1], kernel_size=self._kernels[1][0], padding=self._padding[1], bias=False),
            nn.BatchNorm2d(self._out_ch[1]),
            QuantizeActivation(bits=self._out_bits[1]),
            nn.MaxPool2d(self._kernels[1][1]),

            # Block 2
            QuantConv2d(self._in_ch[2], self._out_ch[2], kernel_size=self._kernels[2][0], padding=self._padding[2], bias=False),
            nn.BatchNorm2d(self._out_ch[2]),
            QuantizeActivation(bits=self._out_bits[2]),
            nn.MaxPool2d(self._kernels[2][1]),

            # Block 3
            QuantConv2d(self._in_ch[3], self._out_ch[3], kernel_size=self._kernels[3][0], padding=self._padding[3], bias=False),
            nn.BatchNorm2d(self._out_ch[3]),
            QuantizeActivation(bits=self._out_bits[3]),
        )

        # Classifier Block
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.1),
            QuantConv2d(self._in_ch[4], self._out_ch[4], kernel_size=self._kernels[4][0], bias=False),
            nn.BatchNorm2d(self._out_ch[4])
        )

        self.render_verilog(out_file)

    def forward(self, x):
        x = self.features(x)           
        
        # Global Max
        # Reduces (Batch, Channels, H, W) -> (Batch, Channels, 1, 1)
        x = torch.amax(x, dim=(2, 3), keepdim=True) 
        
        # Classifier
        # Now the 1x1 Conv acts exactly like your SystemVerilog FC layer
        x = self.classifier(x)         
        
        # 3. Flatten the final output to (Batch, Num_Classes)
        x = torch.flatten(x, 1)
        
        return x