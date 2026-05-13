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
            self._term_bits   = int(dut.TermBits.value)
            self._term_count  = int(dut.TermCount.value)
            # Linear Parameters
            self._in_channels = int(dut.InChannels.value)
            self._class_count = int(dut.ClassCount.value)
            self._weight_bits = int(dut.WeightBits.value)
            unsigned_obj = getattr(dut, "Unsigned", None)
            self._Unsigned = int(unsigned_obj.value) if unsigned_obj is not None else 0
            shift_obj = getattr(dut, "ShiftBits", None)
            self._ShiftBits = int(shift_obj.value) if shift_obj is not None else 0
        else:
            try:
                self._term_bits   = int(kwargs["term_bits"])
                self._term_count  = int(kwargs["term_count"])
                self._in_channels = int(kwargs["in_channels"])
                self._class_count = int(kwargs["class_count"])
                self._weight_bits = int(kwargs.get("weight_bits", 2))
                self._Unsigned    = int(kwargs.get("unsigned_in", 0))
                self._ShiftBits   = int(kwargs.get("ShiftBits", 0))
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
        self._sequence_buffer: List[List[int]] = []

        # Calculate hardware-accurate accumulation width (matches RTL)
        def acc_width(in_b, w_b, ic, b_b):
            max_in = 1 if in_b <= 2 else (1 << (in_b - 1))
            max_w  = 1 if w_b <= 2 else (1 << (w_b - 1))
            worst_case_sum = max_in * max_w * ic
            
            wc_bits = worst_case_sum.bit_length() + 1
            wc_bits = max(wc_bits, b_b) + 1
            return 32

        # Linear layer internally extends InBits to InBits+1 for the signed MAC
        self._acc_bits = acc_width(self._term_bits + 1, self._weight_bits, self._in_channels, int(kwargs.get("bias_bits", 4)) if dut is None else int(dut.BiasBits.value))

    def _truncate_to_bits(self, val: int, bits: int) -> int:
        """Simulates hardware signed truncation."""
        mask = (1 << bits) - 1
        wrapped = val & mask
        sign_bit = 1 << (bits - 1)
        if wrapped >= sign_bit:
            return wrapped - (1 << bits)
        return wrapped

    def _remap_value(self, v: int, bits: int, is_unsigned: bool = False) -> int:
        """Remaps binary/ternary values to their numerical representation.
        For bits==1 (BNN): raw packed bit is 0 or 1 (unpack_terms never
        sign-extends single bits), so map 1->+1, 0->-1.
        For bits==2 (ternary): unpack_terms sign-extends, so values are {-1,0,1}.
        is_unsigned: if True, treat multi-bit values as unsigned (for post-ReLU activations).
        """
        if bits == 1:
            return 1 if v == 1 else -1
        if bits == 2:
            return 1 if v == 1 else (-1 if v == -1 else 0)
        
        if is_unsigned:
             return v & ((1 << bits) - 1)
             
        return v

    def step(self, x: List[int]) -> Optional[List[tuple]]:
        """
        Process a single input vector. Buffers until term_count is reached,
        then performs the classification.
        Returns [(class_id, logits)] on completion, otherwise None.
        """
        # 1. Map values for 1-bit or 2-bit logic before buffering
        # Inputs use the Unsigned flag; weights are always signed.
        current_vector = [self._remap_value(v, self._term_bits, is_unsigned=bool(self._Unsigned)) for v in x]
        self._sequence_buffer.append(current_vector)
        self._term_counter += 1

        # 2. When the full image sequence is collected
        if self._term_counter == self._term_count:
            
            # Remap weights for MAC calculation (always signed, never unsigned)
            remapped_w = np.array([[self._remap_value(int(self.w[oc][ic]), self._weight_bits, is_unsigned=False) 
                                   for ic in range(self._in_channels)] 
                                  for oc in range(self._class_count)])

            # --- BRAIN A: Torch Reference ---
            torch_id, torch_logits = torch_classifier_ref(
                self._sequence_buffer, remapped_w, self.b, 
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
                    acc += int(remapped_w[oc][ic]) * int(manual_max[ic])
                
                # Truncate to match hardware accumulator width (32 bits for LinearBits)
                usable_acc_signed = self._truncate_to_bits(acc, self._acc_bits)
                
                # Apply ShiftBits (Arithmetic Right Shift)
                shifted = usable_acc_signed >> self._ShiftBits
                
                # Store final 32-bit value passed to comparator tree
                manual_logits.append(self._truncate_to_bits(shifted, 32))
            
            self._last_logits = manual_logits
            # Tie-breaking: hardware comparator tree picks the first (lowest) index on ties
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

        # Read hardware logits from the linear layer output for debugging
        try:
            ll = self._dut.dut.linear_layer_inst
            raw = int(ll.data_o.value)
            bits = 32  # LinearBits
            hw_logits = []
            for ch in range(self._class_count):
                val = (raw >> (ch * bits)) & ((1 << bits) - 1)
                if val >= (1 << (bits - 1)):
                    val -= (1 << bits)
                hw_logits.append(val)
            print(f"DEBUG HW LOGITS: {hw_logits}")
        except Exception as e:
            print(f"DEBUG HW LOGITS: could not read ({e})")

        if sim_verbose():
            print(
                f"Produced class {got_id}, expected {expected_id} "
                f"at time {get_sim_time(units='ns')}ns"
            )

        print(f"DEBUG MODEL LOGITS: {self._last_logits}"); assert got_id == expected_id, (
            f"Class mismatch. Expected {expected_id}, got {got_id}"
        )