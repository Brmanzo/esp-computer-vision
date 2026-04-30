# quantize.py

from matplotlib.pylab import cast
import torch
from   torch import nn
import torch.nn.functional as F


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

class QuantConv2d(nn.Conv2d): # Or FusedQuantConvBN2d depending on what you named it
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bits=4, **kwargs):
        # Safely remove 'bias' from kwargs if the user passed it in
        kwargs.pop('bias', None) 
        
        # Now super() only gets one bias argument!
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs)
        
        self._bits = bits
        self._quantize = False
        
        # Native BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_channels))  # Gamma
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))   # Beta
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.eps = 1e-5
        self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. The Dummy Pass: Standard float convolution to get pre-BN activations
        #    (We use None for bias because BN handles all biasing)
        pre_act = F.conv2d(x, self.weight, None, self.stride, self.padding)
        
        # 2. Handle BatchNorm Statistics on the OUTPUT of the convolution
        if self.training:
            # Calculate batch stats on the 16 output channels
            mean = pre_act.mean([0, 2, 3])
            var = pre_act.var([0, 2, 3], unbiased=False)
            
            # Update running stats
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # 3. Calculate the BN Scale
        bn_scale = self.bn_weight / torch.sqrt(var + self.eps)

        # 4. Fold BN into the weights
        # Broadcast scale to (OutChannels, 1, 1, 1)
        folded_w = self.weight * bn_scale.view(-1, 1, 1, 1)
        
        # 5. Calculate the Folded Bias
        folded_b = self.bn_bias - (mean * bn_scale)

        # 6. Quantize the FOLDED weights (The QAT Schedule kicks in here!)
        if self._quantize:
            w_q = cast(torch.Tensor, QuantizeBit.apply(folded_w, self._bits))
        else:
            w_q = folded_w

        # 7. Execute the REAL Convolution using the integer weights and folded bias
        return F.conv2d(x, w_q, folded_b, self.stride, self.padding)

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