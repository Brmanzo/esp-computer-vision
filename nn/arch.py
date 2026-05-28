# nn.architecture.py
# Bradley Manzo 2026

import torch

from nn.config   import NNConfig, ConvConfig, PoolConfig
from nn.globals  import NN_CFG, BRAM_CAP, DSP_CAP, CLK_FREQ_HZ
from nn.quantize import QuantConv2d, QuantizeActivation, LearnedShiftQuantizer

class cnn(torch.nn.Module):
    '''Construct the Pytorch CNN based on provided NNConfig.'''
    def __init__(self, config: NNConfig):
        super().__init__()
        self.config = config
        self._is_fine_tuning = False

        # 1. Dynamically Build the Feature Extractor
        feature_layers: list[torch.nn.Module] = []
        self.cls_input_shift_quantizer: LearnedShiftQuantizer  # set below

        # Iterate naturally over all feature layers in network config
        num_feature_layers = len(self.config.layers)
        for i, layer_cfg in enumerate(self.config.layers):
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

            # For layers whose hardware output uses the gen_learned_shift path (OutBits > 2),
            # use LearnedShiftQuantizer so training sees the same unsigned ReLU clamp [0, clip_val]
            # that the hardware applies.  Binary (1-bit) and ternary (2-bit) layers use signed
            # QuantizeActivation because their hardware paths (gen_binary_out / gen_ternary_out)
            # preserve sign information.
            if conv._out_bits > 2: 
                learned_q = LearnedShiftQuantizer(
                    init_shift = conv._shift,
                    out_bits   = conv._out_bits,
                    min_shift  = max(0, conv._shift - 4),
                    max_shift  = conv._shift + 4,
                )
                if i == num_feature_layers - 1:
                    self.cls_input_shift_quantizer = learned_q
                feature_layers.append(learned_q)
            else:
                feature_layers.append(QuantizeActivation(
                    bits=conv._out_bits,
                    learnable=(conv._out_bits > 1),
                    shift=conv._shift
                ))

            if pool is not None:
                if pool._mode == 0:
                    feature_layers.append(torch.nn.MaxPool2d(pool._kernel_width))
                else:
                    feature_layers.append(torch.nn.AvgPool2d(pool._kernel_width))

        self.features = torch.nn.Sequential(*feature_layers)

        # 2. Build the Classifier Block
        cls_cfg = config.classifier_config

        # Classifier outputs raw logits — argmax selects the class.
        # No output quantizer needed here; the activation quantizer on the
        # last feature conv (cls_input_shift_quantizer) handles the precision.
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout2d(p=0.1),
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
        from nn.verilog   import render_verilog
        render_verilog(self.config)

    def forward(self, x):
        # Map 1-bit input from {0.0, 1.0} to {-1.0, 1.0} to match the hardware's 1-bit signed logic
        x = 2.0 * x - 1.0
        x = self.features(x)           
        
        # Global Max
        # Reduces (Batch, Channels, H, W) -> (Batch, Channels, 1, 1)
        x = torch.amax(x, dim=(2, 3), keepdim=True) 
        
        # Classifier
        x = self.classifier(x)         
        
        # Flatten the final output to (Batch, Num_Classes)
        x = torch.flatten(x, 1)

        return x

    @property
    def hardware_shift(self) -> int:
        '''Returns the learned integer right-shift applied to the classifier input
        activation (last feature conv output). Used by RTL export to parameterize
        the barrel-shift in output_encoder / classifier_layer.
        '''
        return self.cls_input_shift_quantizer.hardware_shift

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, QuantConv2d):
            # Check if it's a low-bit layer to use more conservative initialization
            bits = getattr(m, '_weight_bits', 8)
            actual_bits = bits if isinstance(bits, int) else getattr(bits, '_q_min_bits', 8)
            
            if actual_bits <= 2:
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
            else:
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


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
        rams = BRAM_CAP
        # MultiBufferRAM Usage
        for layer_cfg in self.config.layers:
            rams -= self.rams_per_layer_feature(layer_cfg.ConvLayer)
            if layer_cfg.PoolLayer is not None:
                rams -= self.rams_per_layer_feature(layer_cfg.PoolLayer)

        # Sequential Conv/Linear ROM Usage
        # Each DSP bank cycles through out_ch/dsp_count neurons, so rom_depth divides by dsp_count
        for layer_cfg in self.config.layers:
            if layer_cfg.ConvLayer._dsp_count > 0:
                addr_width = layer_cfg.ConvLayer._dsp_count * layer_cfg.ConvLayer._q_schedule.q_min_bits
                rom_depth  = layer_cfg.ConvLayer._kernel_width**2 * layer_cfg.ConvLayer._in_ch * (layer_cfg.ConvLayer._out_ch // layer_cfg.ConvLayer._dsp_count)
                roms_wide = (addr_width + 15) // 16
                roms_deep = (rom_depth + 255) // 256
                rams -= (roms_wide * roms_deep)

        c_cfg = self.config.classifier_config
        if c_cfg._dsp_count > 0:
            addr_width = c_cfg._dsp_count * c_cfg._q_schedule.q_min_bits
            rom_depth  = c_cfg._in_ch * (c_cfg._num_classes // c_cfg._dsp_count)
            roms_wide  = (addr_width + 15) // 16
            roms_deep  = (rom_depth + 255) // 256
            rams -= (roms_wide * roms_deep)

        assert rams >= 0, f"Network exceeds BRAM budget! Remaining: {rams}"
        used = BRAM_CAP - rams
        utilization = (used / BRAM_CAP) * 100
        print(f"BRAMs: {rams} remaining / {BRAM_CAP} total ({utilization:.1f}% utilization)")
    
    def dsp_utilization(self) -> None:
        dsp = DSP_CAP
        for layer_cfg in self.config.layers:
            if layer_cfg.ConvLayer._dsp_count > 0:
                dsp -= min(layer_cfg.ConvLayer._dsp_count, layer_cfg.ConvLayer._out_ch)
        
        classifier = self.config.classifier_config
        if classifier._dsp_count > 0:
            dsp -= min(classifier._dsp_count, classifier._num_classes)
            
        assert dsp >= 0, f"Network exceeds DSP budget! Remaining: {dsp}"
        used = DSP_CAP - dsp
        utilization = (used / DSP_CAP) * 100
        print(f"DSPs: {dsp} remaining / {DSP_CAP} total ({utilization:.1f}% utilization)")
    
    def cycle_count(self) -> None:
        total_latency = 0
        max_effective_cycles = 0.0
        pixel_decimation = 1 # Tracks how many camera pixels = 1 current layer pixel
        
        print("\n--- PERFORMANCE ANALYSIS ---")
        for i, layer_cfg in enumerate(self.config.layers):
            # Conv Layer
            c_cycles = layer_cfg.ConvLayer._cycle_count
            eff_c = c_cycles / pixel_decimation
            dsp = layer_cfg.ConvLayer._dsp_count
            out_ch = layer_cfg.ConvLayer._out_ch
            dsps_used = min(dsp, out_ch) if dsp > 0 else 0
            print(f"Layer {i} Conv: {c_cycles:>3} cycles ({eff_c:>5.1f} eff) DSPs Used: {dsps_used}")
            
            total_latency += c_cycles
            max_effective_cycles = max(max_effective_cycles, eff_c)
            
            # Pool Layer
            if layer_cfg.PoolLayer is not None:
                p_cycles = layer_cfg.PoolLayer._cycle_count
                eff_p = p_cycles / pixel_decimation
                print(f"Layer {i} Pool: {p_cycles:>3} cycles ({eff_p:>5.1f} eff)")
                
                total_latency += p_cycles
                pixel_decimation *= (layer_cfg.PoolLayer._kernel_width ** 2)
                max_effective_cycles = max(max_effective_cycles, eff_p)
        
        cls = self.config.classifier_config
        cls_cycles = cls._cycle_count
        eff_cls = cls_cycles / pixel_decimation
        dsp = cls._dsp_count
        classes = cls._num_classes
        dsps_used = min(dsp, classes) if dsp > 0 else 0
        linear_cycles = cls._cycle_count - cls._term_count
        print(f"Classifier  : {cls_cycles:>5} cycles ({eff_cls:>5.1f} eff) DSPs Used: {dsps_used}  [GlobalMax: {cls._term_count}, Linear: {linear_cycles}]")
        
        total_latency += cls_cycles
        max_effective_cycles = max(max_effective_cycles, eff_cls)
        
        print(f"\nTotal Pipeline Latency: {total_latency} cycles")
        print(f"Bottleneck (Cycles/Pixel): {max_effective_cycles:.1f}")
        
        if self.config.in_dims.height is not None and self.config.in_dims.width is not None and  max_effective_cycles > 0:
            input_pixels = self.config.in_dims.width * self.config.in_dims.height
            fps = CLK_FREQ_HZ / (max_effective_cycles * input_pixels)
            print(f"Estimated Max Frame Rate: {fps:.2f} FPS")

if __name__ == "__main__":
    network = cnn(NN_CFG)
    print("\n--- Network ARCHITECTURE ---")
    print(network)
    print("\n--- RESOURCE UTILIZATION ---")
    network.ram_utilization()
    network.dsp_utilization()
    network.cycle_count()
    print("")
