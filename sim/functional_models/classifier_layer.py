# classifier_layer.py
import numpy as np
from   typing import List, Optional

from util.utilities  import assert_resolvable, sim_verbose
from util.bitwise    import unpack_terms, pack_terms
from util.torch_ref  import torch_classifier_ref
from util.gen_inputs import gen_input_channels

from   cocotb.utils import get_sim_time


class RandomDataGenerator:
    def __init__(self, dut):
        self._width_p = int(dut.TermBits.value)
        self._InChannels = int(dut.InChannels.value)

    def generate(self):
        raw_din = gen_input_channels(self._width_p, self._InChannels)
        packed_din = pack_terms(raw_din, self._width_p)
        return (packed_din, raw_din)

class ClassifierLayerModel:
    def __init__(self, dut=None, weights: Optional[List[List[int]]]=None, 
                            biases:  Optional[List[int]]=None,
                            **kwargs):
        self._dut = dut

        if dut is not None:
            # Interface Signals
            self._data_i  = dut.data_i
            self._class_o = dut.class_o
            
            # Global Max parameters
            self._term_bits  = int(dut.TermBits.value)
            self._term_count = int(dut.TermCount.value)
            # Linear Parameters
            self._in_channels = int(dut.InChannels.value)
            self._class_count = int(dut.ClassCount.value)
        else:
            try:
                self._term_bits   = int(kwargs["term_bits"])
                self._term_count  = int(kwargs["term_count"])
                self._in_channels = int(kwargs["in_channels"])
                self._class_count = int(kwargs["class_count"])
            except KeyError as e:
                raise ValueError(f"Missing required parameter when dut is None: {e}")

        if weights is None:
            raise ValueError("Weights must be provided to ClassifierLayerModel")
        
        self.w = np.array(weights, dtype=int)
        assert self.w.shape == (self._class_count, self._in_channels), (
            f"Weights shape mismatch: got {self.w.shape}, expected ({self._class_count}, {self._in_channels}). "
        )
        
        if biases is None:
            biases = [0] * self._class_count
            
        self.b = np.array(biases, dtype=int)

        # If scalar bias for OC=1, normalize to shape (1,)
        if self.b.shape == () and self._class_count == 1:
            self.b = self.b.reshape((1,))

        assert self.b.shape == (self._class_count,), (
            f"Bias shape mismatch: got {self.b.shape}, expected ({self._class_count},). "
        )

        self._term_counter = 0
        self._sequence_buffer: list[list[int]] = []

    def step(self, x: List[int]) -> Optional[List[tuple]]:
        """
        Process a single input vector. Buffers until term_count is reached,
        then performs the classification.
        Returns [(class_id, logits)] on completion, otherwise None.
        """
        # 1. Map values for 1-bit logic (0 -> -1) before buffering
        current_vector = [(1 if v == 1 else -1) if self._term_bits == 1 else v for v in x]
        self._sequence_buffer.append(current_vector)
        self._term_counter += 1

        # 2. When the full image sequence is collected
        if self._term_counter == self._term_count:
            
            # --- BRAIN A: Torch Reference ---
            torch_id, torch_logits = torch_classifier_ref(
                self._sequence_buffer, self.w, self.b, 
                self._in_channels, self._class_count
            )

            # --- BRAIN B: Internal Python Model (Manual MAC) ---
            # Manual Max Pool
            manual_max = [max(col) for col in zip(*self._sequence_buffer)]
            
            # Manual Linear Projection
            manual_logits = []
            for oc in range(self._class_count):
                acc = self.b[oc]
                for ic in range(self._in_channels):
                    acc += int(self.w[oc][ic]) * int(manual_max[ic])
                manual_logits.append(acc)
            
            manual_id = manual_logits.index(max(manual_logits))

            # --- CROSS-CHECK A vs B ---
            assert manual_id == int(torch_id), (
                f"Reference Mismatch! Internal Model: {manual_id}, Torch: {torch_id}. "
            )

            # Clean up for next handshake
            self._term_counter = 0
            self._sequence_buffer = []

            # Return the golden data
            return [(int(torch_id), torch_logits)]

        return None

    def consume(self):
        if self._dut is None:
            return None
            
        assert_resolvable(self._data_i)
        packed_in = int(self._data_i.value.integer)
        x = unpack_terms(packed_in, self._term_bits, self._in_channels)
        
        return self.step(x)

    def produce(self, expected):
        if self._dut is None:
            return
            
        assert_resolvable(self._class_o)

        if isinstance(expected, (list, tuple)):
            expected_id = expected[0]
            expected_logits = expected[1] if len(expected) > 1 else None
        else:
            expected_id = expected
            expected_logits = None
        got_id = int(self._class_o.value.integer)

        if sim_verbose():
            print(
                f"Produced class {got_id}, expected {expected_id} "
                f"at time {get_sim_time(units='ns')}ns"
            )

        assert got_id == expected_id, (
            f"Class mismatch. Expected {expected_id}, got {got_id}"
        )