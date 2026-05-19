# cnn.py
# Functional model for the top-level CNN hardware

from typing import List, Optional, Dict, Any
import numpy as np
import os

from functional_models.conv_layer import ConvLayerModel
from functional_models.pool_layer import PoolLayerModel
from functional_models.classifier_layer import ClassifierLayerModel
from nn.config import NNConfig
from util.bitwise import pack_terms

class PictureGenerator:
    def __init__(self, model, inject_pixels: Optional[List[int]] = None) -> None:
        self._width_p = model._InBits
        self._InChannels = model._InChannels

        n_pixels = model._config.in_dims.width * model._config.in_dims.height

        # Priority: explicit inject_pixels arg > INJECT_PIXELS env var > dataset sample
        inject_env = os.environ.get("INJECT_PIXELS", "")
        if inject_pixels is not None:
            self._pixels = inject_pixels
            self._label = None
        elif inject_env == "zeros":
            self._pixels = [0] * n_pixels
            self._label = None
        elif inject_env == "ones":
            self._pixels = [1] * n_pixels
            self._label = None
        elif inject_env:
            self._pixels = [int(v) for v in inject_env.split(",")]
            self._label = None
        else:
            self._sample_idx = int(os.environ.get("SAMPLE_IDX", 0))
            from nn.sample import get_sample
            pixels, label = get_sample(self._sample_idx)
            if pixels is None or label is None:
                raise ValueError(f"Could not load sample {self._sample_idx}")
            self._pixels = pixels
            self._label = label

        self._ptr = 0

    def generate(self) -> tuple[int, List[int]]:
        if self._ptr >= len(self._pixels):
            # If we run out of pixels (shouldn't happen in 1-frame test), return zeros
            raw_din = [0] * self._InChannels
        else:
            # We assume InChannels=1 for these grayscale samples
            raw_din = [self._pixels[self._ptr]]
            self._ptr += 1
            
        packed_din = pack_terms(raw_din, self._width_p)
        return (packed_din, raw_din)

class CNNModel:
    def __init__(self, dut, config: NNConfig, weights_dict: Optional[Dict] = None):
        """
        Instantiates a full functional pipeline based on a NNConfig.
        
        Args:
            dut: The cocotb DUT handle (can be None for standalone use).
            config: The NNConfig instance defining the architecture.
            weights_dict: Optional dictionary containing 'LAYER_N_WEIGHTS' and 'LAYER_N_BIASES'.
                         If None, weights must be provided manually or handled by injection.
        """
        self._dut = dut
        self._config = config
        self._weights_dict = weights_dict or {}

        self.layers: List[Any] = []
        self.results: List[int] = []
        
        # 1. Instantiate Conv/Pool Layers
        for i, layer_cfg in enumerate(config.layers):
            # Conv Layer
            conv_params = self._get_conv_params(layer_cfg.ConvLayer, i)
            conv_model = ConvLayerModel(dut=None, **conv_params)
            self.layers.append(conv_model)
            
            # Optional Pool Layer
            if layer_cfg.PoolLayer:
                pool_params = self._get_pool_params(layer_cfg.PoolLayer, i)
                pool_model = PoolLayerModel(dut=None, **pool_params)
                self.layers.append(pool_model)

        # 2. Instantiate final Classifier Layer
        class_params = self._get_class_params(config.classifier_config, len(config.layers))
        self.classifier = ClassifierLayerModel(dut=None, **class_params)
        self.layers.append(self.classifier)

        # 3. Cache interface parameters
        self._InBits = config._in_bits[0]
        self._InChannels = config.layers[0].ConvLayer._in_ch
        self._deqs = 0

    def _get_conv_params(self, cfg, index):
        w = self._weights_dict.get(f"LAYER_{index}_WEIGHTS")
        b = self._weights_dict.get(f"LAYER_{index}_BIASES")
        is_last = (index == len(self._config.layers) - 1)
        return {
            "KernelWidth": cfg._kernel_width,
            "LineWidthPx": cfg._input_dims.width,
            "LineCountPx": cfg._input_dims.height,
            "InBits":      cfg._in_bits,
            "OutBits":     cfg._out_bits,
            "WeightBits":  cfg._q_schedule._q_min_bits,
            "BiasBits":    cfg._bias_bits,
            "InChannels":  cfg._in_ch,
            "OutChannels": cfg._out_ch,
            "Stride":      cfg._stride,
            "Padding":     cfg._padding,
            "ShiftBits":   self._config.classifier_config._shift if is_last else 0,
            "weights":     w,
            "biases":      b
        }

    def _get_pool_params(self, cfg, index):
        return {
            "KernelWidth": cfg._kernel_width,
            "LineWidthPx": cfg._input_dims.width,
            "LineCountPx": cfg._input_dims.height,
            "InBits":      cfg._in_bits,
            "OutBits":     cfg._out_bits,
            "InChannels":  cfg._in_ch,
            "OutChannels": cfg._out_ch,
            "Stride":      cfg._kernel_width,
            "PoolMode":    cfg._mode,
            "Unsigned":    1 if cfg._in_bits > 2 else 0,
        }

    def _get_class_params(self, cfg, index):
        w = self._weights_dict.get(f"LAYER_{index}_WEIGHTS")
        b = self._weights_dict.get(f"LAYER_{index}_BIASES")
        return {
            "term_bits":   cfg._in_bits,
            "term_count":  cfg._term_count,
            "bus_bits":    cfg._out_bits,
            "in_channels": cfg._in_ch,
            "class_count": cfg._num_classes,
            "weight_bits": cfg._q_schedule._q_min_bits,
            "bias_bits":   cfg._bias_bits,
            "weights":     w,
            "biases":      b,
            "unsigned_in": 1 if cfg._in_bits > 2 else 0,
        }

    def step(self, x: List[int]):
        """
        Chains a single input vector through the entire pipeline.
        Returns a list of final outputs (Class IDs) or None.
        """
        current_burst: List[Any] = [x]
        
        for layer in self.layers:
            next_burst = []
            for item in current_burst:
                # Some layers (Classifier) might receive lists, others (Conv) receive vectors
                # step() logic across components must be consistent.
                # Conv/Pool step() returns List[tuple] or None
                # Classifier step() returns List[tuple] or None
                
                # Special case: Classifier expects a list, not a tuple
                if isinstance(layer, ClassifierLayerModel):
                    res = layer.step(list(item))
                else:
                    res = layer.step(item)
                
                if res is not None:
                    next_burst.extend(res)
            
            current_burst = next_burst
            if not current_burst:
                return None

        # Return flattened results (e.g. [class_id1, class_id2, ...])
        # For classifier, res item is (class_id, logits)
        return [item[0] for item in current_burst]

    def consume(self):
        """Standard ModelRunner interface for top-level DUT."""
        from util.bitwise import unpack_terms
        packed = int(self._dut.data_i.value.integer)
        raw_val = unpack_terms(packed, self._InBits, self._InChannels)
        return self.step(raw_val)

    def produce(self, expected):
        """Standard ModelRunner interface for top-level DUT."""
        from util.utilities import assert_resolvable, sim_verbose
        assert_resolvable(self._dut.data_o)
        got_id = int(self._dut.data_o.value.integer)
        self.results.append(got_id)
        if isinstance(expected, (list, tuple)):
            expected_id = int(expected[0])
        else:
            expected_id = int(expected)

        if sim_verbose():
            print(f"CNN Top-Level Output #{self._deqs}: expected {expected_id}, got {got_id}")

        # Check against PyTorch Inference if requested
        if os.environ.get("CHECK_INFERENCE") == "1":
            sample_idx = int(os.environ.get("SAMPLE_IDX", 0))
            from nn.inference import get_inference
            torch_pred = get_inference(sample_idx)
            if got_id != torch_pred:
                 print(f"\033[93mWARNING: Hardware output ({got_id}) differs from PyTorch prediction ({torch_pred}) for sample {sample_idx}!\033[0m")
            else:
                 print(f"\033[92mSUCCESS: Hardware output matches PyTorch prediction ({got_id}) for sample {sample_idx}!\033[0m")

        assert got_id == expected_id, (
            f"CNN Output Mismatch! Expected {expected_id}, got {got_id} at Output #{self._deqs}"
        )
        self._deqs += 1
