# model.py
# Bradley Manzo 2026

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class QuantizeBit(torch.autograd.Function):
    '''Quantizes weights to a specified number of bits with symmetric quantization.'''
    @staticmethod
    def forward(ctx, w: torch.Tensor, bits: int = 4):
        # Enforce symmetry around zero
        qmin = -2**(bits-1) + 1
        qmax = 2**(bits-1) - 1

        # Compute symmetric scale from max abs weight
        max_abs = w.abs().max()
        scale = max_abs / qmax if max_abs > 0 else 1.0

        q = torch.round(w / scale).clamp(qmin, qmax)
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
            w_q = QuantizeBit.apply(self.weight, self._bits)
            return self._conv_forward(x, w_q, self.bias)
        else:
            return self._conv_forward(x, self.weight, self.bias)

class cnn_model(nn.Module):
    def __init__(self, in_ch=1, num_classes=5):
        super().__init__()

        self._in_ch = [8, 16, 32, 64]

        self.features = nn.Sequential(
            # Block 0
            QuantConv2d(in_ch, self._in_ch[0], kernel_size=3, bias=True),
            nn.BatchNorm2d(self._in_ch[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 1
            QuantConv2d(self._in_ch[0], self._in_ch[1], kernel_size=3, bias=True),
            nn.BatchNorm2d(self._in_ch[1]),
            nn.ReLU(),
            nn.AvgPool2d(2),

            # Block 2
            QuantConv2d(self._in_ch[1], self._in_ch[2], kernel_size=3, bias=True),
            nn.BatchNorm2d(self._in_ch[2]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            QuantConv2d(self._in_ch[2], self._in_ch[3], kernel_size=3, bias=True),
            nn.BatchNorm2d(self._in_ch[3]),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Classifier Block
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.1),
            QuantConv2d(self._in_ch[3], num_classes, kernel_size=1, bias=True, bits=8),
            nn.BatchNorm2d(num_classes)
        )

    def forward(self, x):
        x = self.features(x)           
        x = self.classifier(x)         
        
        # Global MAX pool
        x = torch.amax(x, dim=(2, 3))
        
        return x