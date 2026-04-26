import torch
import torch.nn as nn
import torch.nn.functional as F

def to_torch_input(input_activation):
    # input_activation: [IC][H][W]
    x = torch.tensor(input_activation, dtype=torch.int32)   # (IC,H,W)
    x = x.unsqueeze(0).to(torch.int32)                      # (1,IC,H,W)
    return x

def to_torch_weights(kernels4):
    # kernels4: [OC][IC][K][K]
    w = torch.tensor(kernels4, dtype=torch.int32)           # (OC,IC,K,K)
    return w

def torch_conv_ref(input_activation, kernels4, stride, in_bits=1, out_bits=1):
    x = to_torch_input(input_activation).to(torch.float32)
    w = to_torch_weights(kernels4).to(torch.float32)

    if in_bits == 1:
        # Match MAC input encoding: 0 -> -1, 1 -> +1
        x = x * 2.0 - 1.0

    y = F.conv2d(x, w, stride=stride, padding=0).squeeze(0)

    if out_bits == 1:
        y = (y > 0).to(torch.int32)

    return y

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

def torch_pool_ref(input_activation, kernel_size, stride=None, mode=0):
    x = to_torch_input(input_activation).to(torch.float32)
    if mode == 0:
        y = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=0)
    elif mode == 1:
        y = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=0)
    return y.squeeze(0) 

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