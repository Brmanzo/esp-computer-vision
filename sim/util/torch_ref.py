import torch
import torch.nn as nn
import torch.nn.functional as F

def to_torch_input(input_data):
    """Safely handles both raw lists (from testbench) and Tensors (from previous layers)"""
    if isinstance(input_data, torch.Tensor):
        x = input_data.clone().detach().to(torch.int32)
    else:
        x = torch.tensor(input_data, dtype=torch.int32)
        
    # Add batch dimension if it's exactly 3D (Channels, Height, Width)
    if x.dim() == 3:
        x = x.unsqueeze(0)
        
    return x

def to_torch_weights(kernels4):
    # kernels4: [OC][IC][K][K]
    w = torch.tensor(kernels4, dtype=torch.int32)           # (OC,IC,K,K)
    return w

def torch_conv_ref(input_activation, kernels4, stride, in_bits=1, out_bits=1, padding=0, biases=None):
    x = to_torch_input(input_activation).to(torch.float32)
    w = to_torch_weights(kernels4).to(torch.float32)
    
    # 1. Prepare Biases
    # PyTorch expects a 1D tensor of shape (OutChannels,)
    if biases is not None:
        b = torch.tensor(biases, dtype=torch.float32)
    else:
        # Default to zeros if no bias provided
        b = torch.zeros(w.shape[0], dtype=torch.float32)

    # 2. Input Encoding (Bipolar for 1-bit)
    if in_bits == 1:
        x = x * 2.0 - 1.0
        pad_val = -1.0 
    else:
        pad_val = 0.0

    # 3. Manual Padding
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=pad_val)

    # 4. Run Convolution with Bias
    # We pass 'b' directly into F.conv2d. 
    # It adds the bias to each output channel before returning the result.
    y = F.conv2d(x, w, bias=b, stride=stride, padding=0).squeeze(0)

    if out_bits == 1:
        y = (y > 0).to(torch.int32)
    else:
        # 5. Simulate Hardware Truncation/Sign-Extension
        # We convert to a bitmask to simulate the overflow wrap-around
        y_int = y.to(torch.int32)
        mask = (1 << out_bits) - 1
        y_wrapped = y_int & mask
        
        # 6. Convert back to signed representation for comparison
        sign_bit = 1 << (out_bits - 1)
        y_signed = torch.where(y_wrapped >= sign_bit, y_wrapped - (1 << out_bits), y_wrapped)
        y = y_signed.to(torch.int32)

    return y

def torch_pool_ref(input_activation, kernel_size, stride=None, mode=0):
    x = to_torch_input(input_activation).to(torch.float32)
    if mode == 0:
        y = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=0)
    elif mode == 1:
        y = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=0)
    return y.squeeze(0) 

def torch_single_block_ref(input_activation, kernels4, stride, in_bits=1, out_bits=1, mode=0, pool_kernel_size=2, padding=0):
    # 1. Run Convolution
    conv_out = torch_conv_ref(input_activation, kernels4, stride, in_bits, out_bits, padding)

    # 2. Chain directly into Pool
    pool_out = torch_pool_ref(conv_out, kernel_size=pool_kernel_size, stride=pool_kernel_size, mode=mode)
    
    return pool_out

def torch_classifier_ref(sequence, weights, biases, in_ch, out_ch):
    """
    Performs full pipeline: Global Max Pool -> Linear -> Argmax.
    sequence: Shape (TermCount, InChannels)
    """
    with torch.no_grad():
        # Convert buffer to tensor [TermCount, InChannels]
        t_in = torch.tensor(sequence, dtype=torch.float32)
        
        # 1. Global Max Pool (across the time/sequence dimension)
        # Returns shape [1, InChannels]
        t_max = torch.amax(t_in, dim=0, keepdim=True)

        # 2. Linear Layer
        ref = nn.Linear(in_ch, out_ch, bias=True)
        ref.weight.data = torch.tensor(weights, dtype=torch.float32)
        ref.bias.data = torch.tensor(biases, dtype=torch.float32)
        
        logits = ref(t_max)
        class_id = torch.argmax(logits, dim=1).item()
        
        return class_id, logits.squeeze().tolist()

def torch_linear_ref(weights_2d, biases_1d, InChannels, OutChannels):
    # One output channel instantiates a single neuron
    linear = nn.Linear(in_features=InChannels, out_features=OutChannels, bias=True)

    # Disable gradient tracking
    linear.weight.requires_grad = False
    linear.bias.requires_grad = False

    # Convert to float32 for deterministic conv math
    linear.weight.data = torch.tensor(weights_2d, dtype=torch.float32)
    linear.bias.data   = torch.tensor(biases_1d, dtype=torch.float32)

    return linear