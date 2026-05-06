# model.model.py
# Bradley Manzo 2026

import torch
import torch.nn as nn

from model.config   import ModelConfig, ConvConfig, PoolConfig
from model.quantize import QuantConv2d, QuantizeActivation
from model.globals  import get_hand_gesture_cfg, BRAM_COUNT, DSP_COUNT

class cnn_model(nn.Module):
    '''Construct the Pytorch CNN model based on provided Model Config.'''
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._is_fine_tuning = False

        # 1. Dynamically Build the Feature Extractor
        feature_layers: list[nn.Module] = []
        
        # Iterate naturally over all feature layers in model config
        for layer_cfg in self.config.layers:
            conv = layer_cfg.ConvLayer
            pool = layer_cfg.PoolLayer
            
            feature_layers.append(QuantConv2d(
                in_channels=conv._in_ch, 
                out_channels=conv._out_ch, 
                kernel_size=conv._kernel_width, 
                padding=conv._padding, 
                weight_bits=conv._q_schedule._q_min_bits,
                bias_bits=conv._bias_bits,
                bias=False
            ))
            
            feature_layers.append(QuantizeActivation(
                bits=conv._out_bits,
                learnable=(conv._out_bits <= 2),
                shift=conv._shift
            ))
            
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
            # 1x1 kernel for fully convolutional classifiers
            QuantConv2d(
                in_channels=cls_cfg._in_ch, 
                out_channels=cls_cfg._num_classes, 
                kernel_size=1, 
                weight_bits=cls_cfg._q_schedule._q_min_bits,
                bias_bits=cls_cfg._bias_bits,
                bias=False
            ),
        )
        
        # 2b. Initialize Weights
        self.apply(self._init_weights)

        # 3. Utilities
        self.ram_utilization()
        # Update cnn.sv with current architecture
        from model.render import render_verilog
        render_verilog(self.config)

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

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, QuantConv2d):
            # Check if it's a low-bit layer to use more conservative initialization
            bits = getattr(m, '_weight_bits', 8)
            actual_bits = bits if isinstance(bits, int) else getattr(bits, '_q_min_bits', 8)
            
            if actual_bits <= 2:
                nn.init.normal_(m.weight, mean=0, std=0.01)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def rams_per_layer_feature(self, layer_cfg: ConvConfig | PoolConfig) -> int:
        if layer_cfg._kernel_width == 1:
            return 0
            
        if layer_cfg._input_dims.width is not None:
            # Target bit-width per RAM based on line length
            target_ram_bits = 16 if (layer_cfg._input_dims.width - 1) <= 256 else 8
            
            # Each channel takes (KernelWidth-1)*InBits bits of width in the RAM
            kernel_size = (layer_cfg._kernel_width - 1) * layer_cfg._in_bits
            channels_per_ram = target_ram_bits // kernel_size
            if channels_per_ram == 0: channels_per_ram = 1 # Safety for very large kernels/bits
            
            rams_per_layer = (layer_cfg._in_ch + channels_per_ram - 1) // channels_per_ram
            return rams_per_layer
        return 0

    def ram_utilization(self) -> None:
        rams = BRAM_COUNT
        for layer_cfg in self.config.layers:
            rams -= self.rams_per_layer_feature(layer_cfg.ConvLayer)
            if layer_cfg.PoolLayer is not None:
                rams -= self.rams_per_layer_feature(layer_cfg.PoolLayer)
        
        assert rams >= 0, f"Model exceeds BRAM budget! Remaining: {rams}"
        used = BRAM_COUNT - rams
        utilization = (used / BRAM_COUNT) * 100
        print(f"BRAMs: {rams} remaining / {BRAM_COUNT} total ({utilization:.1f}% utilization)")
    
    def dsp_utilization(self) -> None:
        dsp = DSP_COUNT
        for layer_cfg in self.config.layers:
            dsp -= layer_cfg.ConvLayer._use_dsp * layer_cfg.ConvLayer._out_ch
        
        classifier = self.config.classifier_config
        if classifier._use_dsp == 1:
            dsp -= classifier._num_classes
        elif classifier._use_dsp == 2:
            dsp -= 1
            
        assert dsp >= 0, f"Model exceeds DSP budget! Remaining: {dsp}"
        used = DSP_COUNT - dsp
        utilization = (used / DSP_COUNT) * 100
        print(f"DSPs: {dsp} remaining / {DSP_COUNT} total ({utilization:.1f}% utilization)")

if __name__ == "__main__":
    model = cnn_model(get_hand_gesture_cfg())
    print("\n--- MODEL ARCHITECTURE ---")
    print(model)
    print("\n--- RESOURCE UTILIZATION ---")
    model.ram_utilization()
    model.dsp_utilization()
    print("")
