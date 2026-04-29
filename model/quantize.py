# quantize.py

from matplotlib.pylab import cast
from torch import nn
import torch

class QSchedule:
    def __init__(self, q_start=0, q_epochs=[15,15,15,15,15,15,30], q_max_bits=8, q_min_bits=2):
        self._q_start = q_start
        assert len(q_epochs) == q_max_bits - q_min_bits + 1, "q_epochs must specify epochs for each bit-width quantization step + the final plateau"
        self._epochs_per_bit = q_epochs
        self._q_max_bits = q_max_bits
        self._q_min_bits = q_min_bits

    def total_epochs(self):
        '''Return the total epochs to carry out the final quantization'''
        return self._q_start + sum(self._epochs_per_bit)
    
    def get_target_bits(self, current_epoch):
            '''Returns the target bit-width for a given epoch, or None if quantization hasn't started.'''
            if current_epoch < self._q_start:
                return None
                
            epochs_passed = current_epoch - self._q_start
            accumulated_epochs = 0
            
            # Walk through the schedule to find our current bit-width
            for i, duration in enumerate(self._epochs_per_bit):
                accumulated_epochs += duration
                if epochs_passed < accumulated_epochs:
                    return self._q_max_bits - i
                    
            # If we've passed the final scheduled duration, clamp to min bits
            return self._q_min_bits

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
import torch
import torch.nn as nn

class QuantizeActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, clip_val=None):
        ctx.bits = bits
        ctx.save_for_backward(x, clip_val)

        # 1-Bit Logic (Unchanged)
        if bits == 1:
            return torch.where(x > 0, 1.0, -1.0)

        # 8-Bit Hardware-friendly symmetric quantization
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1

        if clip_val is None:
            raise ValueError("clip_val must be provided when bits > 1")

        # Use the learned clip_val instead of a dynamic batch maximum
        # We clamp it to a tiny minimum to avoid division by zero
        abs_clip = clip_val.abs().clamp(min=1e-4)
        scale = abs_clip / qmax

        # Clamp inputs to the learned hardware range
        x_c = torch.clamp(x, -abs_clip, abs_clip)

        # Quantize (Simulates hardware integer representation)
        q = torch.round(x_c / scale)
        
        # Dequantize (Scales back to float so PyTorch can continue training)
        x_q = q * scale

        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        x, clip_val = ctx.saved_tensors
        bits = ctx.bits

        grad_x = grad_output.clone()
        grad_clip_val = None

        if bits == 1:
            grad_x[x.abs() > 1] = 0
            return grad_x, None, None

        if clip_val is None:
            return grad_x, None, None

        abs_clip = clip_val.abs()

        # STE for Multibit: Cancel gradients ONLY for activations outside our learned range
        grad_x[x.abs() > abs_clip] = 0
        
        # Calculate the gradient for our learnable clipping parameter
        if clip_val is not None and ctx.needs_input_grad[2]:
            grad_clip_val = torch.sum(grad_output * (x > abs_clip).float()) - \
                            torch.sum(grad_output * (x < -abs_clip).float())

        return grad_x, None, grad_clip_val

class QuantizeActivation(nn.Module):
    def __init__(self, bits=1):
        super().__init__()
        self.bits = bits
        if bits > 1:
            # Initialize the learnable clipping threshold. 
            # 3.0 is a great starting point since BatchNorm keeps ~99% of values within [-3, 3]
            self.clip_val = nn.Parameter(torch.tensor(3.0)) 
        else:
            self.clip_val = None

    def forward(self, x):
        return QuantizeActivationSTE.apply(x, self.bits, self.clip_val)