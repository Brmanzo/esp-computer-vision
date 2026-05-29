#!/usr/bin/env python3
# nn.quantize.py
# Bradley Manzo 2026

import math
from   typing import cast, Optional
import torch
from   torch import nn
import torch.nn.functional as F

class QSchedule:
    '''Defines the initial and final quantizations widths, as well as the cycle count to transition between each bit-width during training.'''
    def __init__(self, q_start=0, q_epochs=[15,15,15,15,15,15,30], q_max_bits=8, q_min_bits=2):
        self._q_start = q_start
        assert len(q_epochs) == q_max_bits - q_min_bits + 1, "q_epochs must specify epochs for each bit-width quantization step + the final plateau"
        self._epochs_per_bit = q_epochs
        self._q_max_bits     = q_max_bits
        self._q_min_bits     = q_min_bits

    def total_epochs(self):
        '''Return the total epochs to carry out the final quantization'''
        return self._q_start + sum(self._epochs_per_bit)
    
    def get_target_bits(self, current_epoch):
        '''Returns the target bit-width for a given epoch during training, or None if quantization hasn't started.'''
        if current_epoch < self._q_start:
            return None
            
        epochs_passed = current_epoch - self._q_start
        accumulated_epochs = 0
        # walk through the schedule to find our current bit-width
        for i, duration in enumerate(self._epochs_per_bit):
            accumulated_epochs += duration
            if epochs_passed < accumulated_epochs:
                return self._q_max_bits - i
                
        # If we've passed the final scheduled duration, clamp to min bits
        return self._q_min_bits

    @property
    def q_min_bits(self):
        '''Property to access the minimum (final) bit-width of this schedule.'''
        return self._q_min_bits

def generate_random_quantized_weights(shape: tuple, bits: int) -> torch.Tensor:
    '''Generates a random tensor of weights uniformly distributed within the specified integer bit-width range.'''
    if bits == 2:
        qmin, qmax = -1, 1
    else:
        qmin = -2**(bits-1)
        qmax = 2**(bits-1) - 1
        
    return torch.randint(int(qmin), int(qmax) + 1, shape).float()

class QuantizeWeight(torch.autograd.Function):
    '''Quantizes weights to a specified number of bits with symmetric quantization.'''
    @staticmethod
    def forward(ctx, w: torch.Tensor, bits: int = 4):
        '''Clip current full precision weights to the specified integer range for forward pass.'''
        # Handle the special case of 2-bit ternary quantization {-1, 0, 1} (symmetrical)
        if bits == 2:
            qmin, qmax = -1, 1
        else:
            qmin = -2**(bits-1)
            qmax = 2**(bits-1) - 1

        # Compute symmetric scale from max abs weight
        max_abs = w.abs().max()
        raw_scale = max_abs / qmax if max_abs > 0 else 1.0
        # Force scale to a power of two to match hardware shifts
        scale = 2.0 ** torch.round(torch.log2(raw_scale))

        # Q(x; s, bits) = s * clip(round(x/s), -2^(bits-1), 2^(bits-1)-1)
        q   = torch.round(w / scale)
        q   = torch.clip(q, qmin, qmax)
        w_q = q * scale

        return w_q

    @staticmethod
    def backward(ctx, *grad_outputs):
        '''Backpropagation is straight-through: pass the gradient through unchanged, and return None for bits since it's not a learnable parameter.'''
        grad_output = grad_outputs[0]
        # Return gradient for 'w', and None for 'bits'
        return grad_output, None

class QuantConv2d(nn.Conv2d):
    '''Folds batchnorm into the convolutional weights and biases, then applies quantization to the folded weights. 
    This allows us to train a quantized network with batchnorm effects without needing separate BN layers in hardware.'''
    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_bits=4, bias_bits=8, **kwargs):
        # Safely remove 'bias' from kwargs if the user passed it in
        kwargs.pop('bias', None)

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=True, **kwargs)
        
        self._weight_bits = weight_bits
        self._bias_bits   = bias_bits
        self._quantize    = False
        
        # Native BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_channels))  # Gamma
        self.bn_bias   = nn.Parameter(torch.zeros(out_channels))   # Beta
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.eps       = 1e-5
        self.momentum  = 0.1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 1. The Dummy Pass: Standard float convolution to get pre-BN activations
        #    (None for bias because BN handles all biasing)
        pre_act = F.conv2d(input, self.weight, None, self.stride, self.padding)
        
        # 2. Handle BatchNorm Statistics on the output of the convolution
        if self.training:
            # Calculate batch stats on the output channels
            mean = pre_act.mean([0, 2, 3])
            var  = pre_act.var([0, 2, 3], unbiased=False)
            
            # Update running stats
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean = self.running_mean
            var  = self.running_var

        # 3. Calculate the BN Scale
        bn_scale = self.bn_weight / torch.sqrt(var + self.eps)

        # 4. Fold BN into the weights
        # Broadcast scale to (OutChannels, 1, 1, 1)
        folded_w = self.weight * bn_scale.view(-1, 1, 1, 1)
        
        # 5. Calculate the folded bias
        folded_b = self.bn_bias - (mean * bn_scale)

        # 6. Quantize the folded weights and biases
        if self._quantize:
            # Compute the weight scale matching QuantizeWeight.forward
            if self._weight_bits == 2:
                qmax = 1.0
            else:
                qmax = float(2**(self._weight_bits - 1) - 1)
            max_abs = folded_w.abs().max()
            raw_scale = max_abs / qmax if max_abs > 0.0 else 1.0
            # Force scale to a power of two to match hardware shifts
            w_scale = 2.0 ** torch.round(torch.log2(raw_scale))

            # Quantize weights
            w_q = cast(torch.Tensor, QuantizeWeight.apply(folded_w, self._weight_bits))

            # Quantize biases with STE matching hardware's bias scaling & clamping
            b_min = float(-2**(self._bias_bits - 1))
            b_max = float(2**(self._bias_bits - 1) - 1)
            b_hw = torch.clamp(torch.round(folded_b / w_scale), b_min, b_max)
            folded_b = folded_b + (b_hw * w_scale - folded_b).detach()
        else:
            w_q = folded_w

        # 7. Execute the real convolution using the quantized integer weights and folded bias
        return F.conv2d(input, w_q, folded_b, self.stride, self.padding)

class QuantizeActivationSTE(torch.autograd.Function):
    '''Hybrid STE: Signed/Rounding for 1-2 bits, Unsigned/Flooring for 3+ bits.'''
    @staticmethod
    def forward(ctx, x, bits, clip_val=None, learnable=True, shift=0):
        ctx.bits = bits
        ctx.save_for_backward(x, clip_val)

        # ==========================================
        # PATH A: Signed / Symmetric (1 and 2-bit)
        # Matches your hardware's 'sign' modules
        # ==========================================
        if bits == 1 or bits == 2:
            if bits == 1:
                return torch.where(x > 0, 1.0, -1.0)
            
            # 2-bit Ternary
            qmin, qmax = -1, 1
            if not learnable:
                scale = 2.0 ** shift
                abs_clip = float(qmax) * scale
            else:
                raw_clip = clip_val.abs().clamp(min=1e-4)
                raw_scale = raw_clip / qmax
                pow2_scale = 2.0 ** torch.round(torch.log2(raw_scale))
                abs_clip = raw_clip + (pow2_scale * qmax - raw_clip).detach()
                scale = pow2_scale

            # Signed clamping and Rounding
            x_c = torch.clamp(x, -abs_clip, abs_clip)
            q = torch.round(x_c / scale)
            return q * scale

        # ==========================================
        # PATH B: Unsigned / ReLU (3+ bits)
        # Matches your hardware's 'ReLU>>sh' modules
        # ==========================================
        qmin = 0
        qmax = (2 ** bits) - 1  # Unsigned full range

        if not learnable:
            scale = 2.0 ** shift
            abs_clip = float(qmax) * scale
        else:
            if clip_val is None:
                raise ValueError("clip_val must be provided when bits > 1 and learnable=True")
            raw_clip = clip_val.abs().clamp(min=1e-4)
            raw_scale = raw_clip / qmax
            pow2_scale = 2.0 ** torch.round(torch.log2(raw_scale))
            abs_clip = raw_clip + (pow2_scale * qmax - raw_clip).detach()
            scale = pow2_scale

        # Unsigned clamping (ReLU) and Flooring (Right-Shift)
        x_c = torch.clamp(x, 0.0, abs_clip)
        q = torch.floor(x_c / scale)
        return q * scale

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        x, clip_val = ctx.saved_tensors
        bits = ctx.bits

        grad_x = grad_output.clone()
        grad_clip_val = None

        # Backward for 1-bit and 2-bit (Original User Logic)
        if bits == 1 or bits == 2:
            grad_x[x.abs() > 1] = 0
            return grad_x, None, None, None, None

        if clip_val is None:
            return grad_x, None, None, None, None

        abs_clip = clip_val.abs()

        # Backward for 3+ bits (Unsigned ReLU Logic)
        grad_x[(x > abs_clip) | (x < 0)] = 0
        
        if clip_val is not None and ctx.needs_input_grad[2]:
            grad_clip_val = torch.sum(grad_output * (x > abs_clip).float())

        return grad_x, None, grad_clip_val, None, None

class QuantizeActivation(nn.Module):
    '''Quantizes activations to a specified bit-width.'''
    def __init__(self, bits=1, learnable=True, shift=0):
        super().__init__()
        self.bits = bits
        self.learnable = learnable
        self.shift = shift
        self.clip_val: Optional[nn.Parameter]
        if bits > 1 and learnable:
            # Initialize the learnable clipping threshold. 
            # Ternary (bits=2) needs a smaller threshold than multi-bit to avoid a huge dead zone.
            init_val = 1.0 if bits == 2 else 3.0
            self.clip_val = nn.Parameter(torch.tensor(init_val)) 
        else:
            self.clip_val = None

    def forward(self, input):
        '''Return Quantized Straight Through Estimator output for activations.'''
        return QuantizeActivationSTE.apply(input, self.bits, self.clip_val, self.learnable, self.shift)


class LearnedShiftQuantizer(nn.Module):
    '''Unsigned learned-range quantizer for the classifier input activation.

    During training: learns clip_val in the BN-normalised float domain (same
    mechanism as QuantizeActivation with learnable=True) using unsigned ReLU
    clamping [0, clip_val] and an STE rounding pass.  Gradients flow cleanly
    through to all preceding layers.

    On export: derives the integer hardware barrel-shift from clip_val and the
    known full-precision accumulator width (acc_bits = init_shift + out_bits).
    clip_val is also surfaced as act_scale so export.py propagates it into the
    classifier weight-compensation path, exactly as for QuantizeActivation.

    Args:
        init_shift:  Analytically-derived shift (acc_bits - out_bits).  Used
                     only to record acc_bits; does NOT initialise the scale.
        out_bits:    Unsigned output bit-width (e.g. 4 -> range [0, 15]).
        min_shift:   Lower bound clamped on hardware_shift (default 0).
        max_shift:   Upper bound clamped on hardware_shift (default init_shift+4).
    '''
    def __init__(self, init_shift: int, out_bits: int,
                 min_shift: int = 0, max_shift: Optional[int] = None):
        super().__init__()
        self._out_bits  = out_bits
        self._acc_bits  = init_shift + out_bits  # full integer accumulator width
        self._min_shift = min_shift
        self._max_shift = max_shift if max_shift is not None else init_shift + 4
        # Learnable upper clipping boundary in BN-normalised float space.
        # 3.0 matches QuantizeActivation(learnable=True) default and keeps
        # gradients from being crushed on the first epoch.
        self.clip_val = nn.Parameter(torch.tensor(3.0))

    def _pow2_clip(self) -> torch.Tensor:
        '''Nearest power-of-2 clip value (STE-compatible tensor).
        Rounding log2(clip_val) to the nearest integer constrains the clip boundary
        to {…, 0.25, 0.5, 1, 2, 4, 8, …}, making hardware_shift an exact integer
        and eliminating the floor-approximation error in the float→integer conversion.
        '''
        raw = self.clip_val.abs().clamp(min=1e-4)
        pow2 = 2.0 ** torch.round(torch.log2(raw))
        return raw + (pow2 - raw).detach()   # STE: pow2 in forward, gradient through raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qmax     = float(2 ** self._out_bits - 1)   # unsigned: 15 for 4-bit
        abs_clip = self._pow2_clip()
        scale    = abs_clip / qmax

        # --- Unsigned ReLU clamp [0, clip_val] ---
        x_c = x.clamp(min=0.0).clamp(max=abs_clip)

        # --- Quantise with STE: floor in forward (matches hardware shift), straight-through gradient ---
        q     = torch.floor(x_c / scale)
        x_out = x_c + (q * scale - x_c).detach()
        return x_out

    @property
    def act_scale(self) -> float:
        '''Float-domain scale = clip_pow2 / qmax, using the power-of-2 constrained clip.'''
        qmax = 2 ** self._out_bits - 1
        clip = float(self._pow2_clip().item())
        return clip / qmax

    @property
    def hardware_shift(self) -> int:
        '''Analytical barrel-shift estimate: acc_bits - out_bits adjusted by clip ratio.

        NOTE: this is an architectural estimate only — it ignores the actual trained
        weight scale (w_scale).  Export uses the empirical formula
        floor(log2(clip_pow2 / (qmax * w_scale))) instead, which accounts for the
        real weight magnitudes.  Use this property only for display or rough checks.
        '''
        base_shift = self._acc_bits - self._out_bits
        clip       = float(self._pow2_clip().item())
        adjustment = round(math.log2(max(clip, 1e-4) / 3.0))
        s = base_shift - adjustment
        return max(self._min_shift, min(self._max_shift, s))