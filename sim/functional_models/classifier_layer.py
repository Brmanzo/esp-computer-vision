# classifier_layer.py
import numpy as np
from   typing import List

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
    def __init__(self, dut, weights: List[List[int]], biases: List[int]):
        self._dut = dut
        self._data_i = dut.data_i
        self._class_o = dut.class_o

        # Global Max parameters
        self._term_bits = int(dut.TermBits.value)
        self._term_count = int(dut.TermCount.value)
        self._term_counter = 0
        self._current_max = None

        # Linear Parameters
        self._in_channels  = int(dut.InChannels.value)
        self._class_count = int(dut.ClassCount.value)

        # 2D array storing all weights in each filter: [OC][IC]
        self.w = np.array(weights, dtype=int)
        assert self.w.shape == (self._class_count, self._in_channels)

        # 1D array storing bias for each output channel: [OC]
        self.b = np.array(biases, dtype=int)

        # If scalar bias for OC=1, normalize to shape (1,)
        if self.b.shape == () and self._class_count == 1:
            self.b = self.b.reshape((1,))

        assert self.b.shape == (self._class_count,), (
            f"Bias shape mismatch: got {self.b.shape}, expected ({self._class_count},). "
            f"biases={biases!r}"
        )

        self._sequence_buffer: list[list[int]] = []

    def consume(self):
        assert_resolvable(self._data_i)
        packed_in = int(self._data_i.value.integer)
        x = unpack_terms(packed_in, self._term_bits, self._in_channels)

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
                    acc += self.w[oc][ic] * manual_max[ic]
                manual_logits.append(acc)
            
            manual_id = manual_logits.index(max(manual_logits))

            # --- CROSS-CHECK A vs B ---
            # Use math.isclose or np.allclose if using floats, 
            # but for integers, exact equality is expected.
            assert manual_id == torch_id, (
                f"Reference Mismatch! Internal Model: {manual_id}, Torch: {torch_id}. "
                f"Check for tie-breaking or precision drift."
            )

            # Clean up for next handshake
            self._term_counter = 0
            self._sequence_buffer = []

            # Return the golden data for produce() to check against BRAIN C (the DUT)
            return [(torch_id, torch_logits)]

        return None

    def produce(self, expected):
        assert_resolvable(self._class_o)

        expected_id, expected_logits = expected
        got_id = int(self._class_o.value.integer)

        if sim_verbose():
            print(
                f"Produced class {got_id}, expected {expected_id}"
                f"logits={expected_logits} at time {get_sim_time(units='ns')}ns"
            )

        assert got_id == expected_id, (
            f"Class mismatch. Expected {expected_id}, got {got_id}"
        )