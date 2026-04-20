# model.py
# Bradley Manzo 2026

import torch
import torch.nn as nn
from typing import cast

import torch
import torch.nn as nn

from pathlib import Path

BRAM_COUNT = 30 - 1 # Subtract 1 for the Skid Buffer BRAM on deframer

from .render import render_header, render_wires, render_conv_layer, render_pool_layer, render_footer

class QuantizeBit(torch.autograd.Function):
    '''Quantizes weights to a specified number of bits with symmetric quantization.'''
    @staticmethod
    def forward(ctx, w: torch.Tensor, bits: int = 4):
        if bits == 2:
            qmin, qmax = -1, 1
        else:
            qmin = -2**(bits-1)
            qmax = 2**(bits-1) - 1

        # Compute symmetric scale from max abs weight
        max_abs = w.abs().max()
        scale = max_abs / qmax if max_abs > 0 else 1.0

        # Q(x;s=scale,b=bits) = s * clip(round(x/s), (-2^b), (2^b)-1)
        q = torch.round(w / scale)
        q = torch.clip(q, qmin, qmax)
        w_q = q * scale

        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        # Return gradient for 'w', and None for 'bits'
        return grad_output, None

class QuantConv2d(nn.Conv2d):
    '''Applies quantization to conv2d layer weights during the forward pass when enabled.'''
    def __init__(self, *args, threshold: float = 0.05, bits: int = 4, **kwargs):
        # Hardware tip: If your FPGA/ASIC doesn't support biases yet, 
        # force bias=False here or pass it in kwargs.
        super().__init__(*args, **kwargs)
        self._threshold = threshold
        self._bits = bits
        self._quantize = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._quantize:
            # Cast the output to guarantee to the linter that this is a Tensor
            w_q = cast(torch.Tensor, QuantizeBit.apply(self.weight, self._bits))
            return self._conv_forward(x, w_q, self.bias)
        else:
            return self._conv_forward(x, self.weight, self.bias)
        
class BinaryActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x > 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = grad_output.clone()
        grad[x.abs() > 1] = 0
        return grad
    
class BinaryActivation(nn.Module):
    def forward(self, x):
        return BinaryActivationSTE.apply(x)

class cnn_model(nn.Module):

    def ram_utilization(self) -> None:
        rams = BRAM_COUNT
        # Each BRAM can support 4 input channels
        for layer in range(len(self._kernels)):
            for module in range(len(self._kernels[layer])):
                if (self._kernels[layer][module] - 1) <= 0:
                    continue # Skip layers with kernel size 1 (e.g., classifier)
                target_ram_bits  = 16 if (self._input_widths[layer][module] - 1) <= 256 else 8
                channels_per_ram = target_ram_bits // (self._kernels[layer][module] - 1) * self._input_bits[layer][module]
                rams_per_layer = (self._in_ch[layer] + channels_per_ram - 1) // channels_per_ram
                
                rams -= rams_per_layer
        
        assert rams >= 0, f"Model exceeds BRAM budget! Remaining: {rams}"
        print(f"BRAMs remaining after model layers: {rams}")

    def render_verilog(self, filepath: Path) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            print(render_header(), file=f)
            print(render_wires(self._kernels), file=f)
            print("", file=f)

            for layer in range(self._layers):
                for module in range(len(self._kernels[layer])):
                    if module == 0:  # Convolution module
                        print(
                            render_conv_layer(
                                LineWidthPx=self._input_widths[layer][0],
                                LineCountPx=self._input_heights[layer][0],
                                InBits=self._input_bits[layer][0],
                                OutBits=self._output_bits[layer],
                                KernelWidth=self._kernels[layer][0],
                                WeightBits=self._weight_bits[layer],
                                InChannels=self._in_ch[layer],
                                OutChannels=self._in_ch[layer + 1] if layer < self._layers - 1 else self._num_classes,
                                Stride=self._stride[layer],
                                Weights=f"weights_{layer}",
                                instance=layer,
                                num_instances=self._layers,
                                kernels=self._kernels,
                            ),
                            file=f,
                        )
                    elif module == 1:  # Pooling module
                        print(
                            render_pool_layer(
                                LineWidthPx=self._input_widths[layer][1],
                                LineCountPx=self._input_heights[layer][1],
                                InBits=self._input_bits[layer][1],
                                KernelWidth=self._kernels[layer][1],
                                InChannels=self._in_ch[layer + 1] if layer < self._layers - 1 else self._num_classes,
                                instance=layer,
                                num_instances=self._layers,
                                kernels=self._kernels,
                            ),
                            file=f,
                        )

            print(render_footer(), file=f)

    def __init__(self, input_dimensions, in_channels, kernels, schedule, num_classes=5):
        super().__init__()

        repo_root = Path(__file__).resolve().parents[1]   # adjust depth as needed
        out_file = repo_root / "rtl" / "blocks" / "cnn.sv"

        self._in_ch       = in_channels
        self._num_classes = num_classes
        self._layers = len(self._in_ch)
        
        self._input_bits       = [[1, 1] for _ in range(self._layers)]
        self._output_bits      = [    1  for _ in range(self._layers)]
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
                    w = w - self._kernels[layer][module] + 1
                    h = h - self._kernels[layer][module] + 1

                elif module == 1: # Pooling module
                    w = w // self._kernels[layer][1]
                    h = h // self._kernels[layer][1]

        self._weight_bits   = [schedule[i]._q_min_bits for i in range(self._layers)]

        self._stride        = [    1  for _ in range(self._layers)]
        
        # One BRAM consumed by binary in_ch
        self.ram_utilization()

        self.render_verilog(out_file)

        self.features = nn.Sequential(
            # Block 0
            QuantConv2d(self._in_ch[0], self._in_ch[1], kernel_size=self._kernels[0][0], padding=0, bias=False),
            nn.BatchNorm2d(self._in_ch[1]),
            BinaryActivation(),
            nn.MaxPool2d(self._kernels[0][1]),

            # Block 1
            QuantConv2d(self._in_ch[1], self._in_ch[2], kernel_size=self._kernels[1][0], padding=0, bias=False),
            nn.BatchNorm2d(self._in_ch[2]),
            BinaryActivation(),
            nn.MaxPool2d(self._kernels[1][1]),

            # Block 2
            QuantConv2d(self._in_ch[2], self._in_ch[3], kernel_size=self._kernels[2][0], padding=0, bias=False),
            nn.BatchNorm2d(self._in_ch[3]),
            BinaryActivation(),
            nn.MaxPool2d(self._kernels[2][1]),

            # Block 3
            QuantConv2d(self._in_ch[3], self._in_ch[4], kernel_size=self._kernels[3][0], padding=0, bias=False),
            nn.BatchNorm2d(self._in_ch[4]),
        )

        # Classifier Block
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.1),
            QuantConv2d(self._in_ch[4], self._num_classes, kernel_size=self._kernels[4][0], bias=False),
            nn.BatchNorm2d(self._num_classes)
        )

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