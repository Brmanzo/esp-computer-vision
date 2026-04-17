# model.py
# Bradley Manzo 2026

import torch
import torch.nn as nn
from typing import cast

import torch
import torch.nn as nn

BRAM_COUNT = 29

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
    def __init__(self, in_ch=1, num_classes=5):
        super().__init__()

        self._in_ch = [8, 16, 24, 32]
        
        # One BRAM consumed by binary in_ch
        rams = BRAM_COUNT - in_ch
        # Each BRAM can support 4 input channels
        for ch in self._in_ch:
            rams -= (ch // 8)   # Cost per conv_layer
            rams -= (ch // 16)  # Cost per pool_layer
        
        assert rams >= 0, f"Model exceeds BRAM budget! Remaining: {rams}"
        print(f"BRAMs remaining after model layers: {rams}")

        self.features = nn.Sequential(
            # Block 0
            QuantConv2d(in_ch, self._in_ch[0], kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(self._in_ch[0]),
            BinaryActivation(),
            nn.MaxPool2d(2),

            # Block 1
            QuantConv2d(self._in_ch[0], self._in_ch[1], kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(self._in_ch[1]),
            BinaryActivation(),
            nn.MaxPool2d(2),

            # Block 2
            QuantConv2d(self._in_ch[1], self._in_ch[2], kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(self._in_ch[2]),
            BinaryActivation(),
            nn.MaxPool2d(2),

            # Block 3
            QuantConv2d(self._in_ch[2], self._in_ch[3], kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(self._in_ch[3]),
        )

        # Classifier Block
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.1),
            QuantConv2d(self._in_ch[3], num_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_classes)
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