# linear_layer.py
import numpy as np
from   typing import List

from util.utilities  import sim_verbose
from util.bitwise    import sign_extend, unpack_terms, pack_terms
from util.gen_inputs import gen_input_channels

def output_width(width_in: int, weight_width: int, bias_bits: int, in_channels: int=1) -> str:
    '''Calculates proper output width for given input width amount of accumulations.'''
    terms = in_channels

    max_val    = (1 << width_in) - 1       # Unsigned
    max_weight = (1 << (weight_width - 1))

    max_sum = terms * max_val * max_weight
    abs_bits = max_sum.bit_length()
    if bias_bits > abs_bits:
        return str(bias_bits + 1)
    return str(abs_bits + 1)   # +1 for sign bit

    
class RandomDataGenerator:
    def __init__(self, dut):
        self._width_p = int(dut.InBits.value)
        self._InChannels = int(dut.InChannels.value)

    def generate(self):
        din = gen_input_channels(self._width_p, self._InChannels)
        packed = pack_terms(din, self._width_p)
        return packed, din


class LinearLayerModel():
    def __init__(self, dut, weights: List[List[int]], biases: List[int], torch_ref=None):
        self._dut = dut
        self._InBits  = int(dut.InBits.value)
        self._OutBits = int(dut.OutBits.value)
        self._InChannels  = int(dut.InChannels.value)
        self._OutChannels = int(dut.OutChannels.value)

        self.w = np.array(weights, dtype=int)
        self.b = np.array(biases, dtype=int)
        self._torch_ref = torch_ref # Optional nn.Linear reference

    def consume(self):
        """Called by ModelRunner on input handshake."""
        # Unpack bits from pins using generic utility
        packed = int(self._dut.data_i.value)
        raw_inputs = unpack_terms(packed, self._InBits, self._InChannels)

        # 1. Handle BNN mapping if InBits == 1
        if self._InBits == 1:
            input_vals = [1 if x == 1 else -1 for x in raw_inputs]
        else:
            input_vals = raw_inputs

        # 2. Compute expected output (Standard Math)
        expected = []
        for oc in range(self._OutChannels):
            acc = int(self.b[oc])
            for ic in range(self._InChannels):
                acc += int(self.w[oc][ic]) * input_vals[ic]
            expected.append(acc)

        if self._OutBits == 1:
            expected = [1 if x > 0 else 0 for x in expected]
        return [expected]

    def produce(self, expected):
        """Called by ModelRunner on output handshake."""
        got_raw = unpack_terms(int(self._dut.data_o.value), self._OutBits, self._OutChannels)
        
        for ch in range(self._OutChannels):
            got = got_raw[ch]
            if self._OutBits == 1:
                exp = expected[ch]
            else:
                exp = sign_extend(expected[ch] & ((1 << self._OutBits) - 1), self._OutBits)
            if sim_verbose():
                print(f"Output ch{ch}: expected {exp}, got {got}")
            assert got == exp, f"Mismatch ch{ch}: expected {exp}, got {got}"