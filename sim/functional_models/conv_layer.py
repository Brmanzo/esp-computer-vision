import numpy as np
from   pathlib import Path
import queue
from   typing import List, Optional

from util.utilities  import assert_resolvable
from util.bitwise    import sign_extend, pack_terms, unpack_terms
from util.gen_inputs import gen_input_channels

def output_width(width_in: int, weight_width: int, kernel_width: int=3, in_channels: int=1) -> str:
    terms = kernel_width * kernel_width * in_channels

    # For signed inputs, the max magnitude is 2^(width-1)
    max_val    = (1 << (width_in - 1)) 
    max_weight = (1 << (weight_width - 1))

    max_sum = terms * max_val * max_weight
    abs_bits = max_sum.bit_length()
    return str(abs_bits + 1)

class RandomDataGenerator:
    def __init__(self, dut):
        self._width_p = int(dut.InBits.value)
        self._InChannels = int(dut.InChannels.value)

    def generate(self):
        raw_din = gen_input_channels(self._width_p, self._InChannels)
        packed_din = pack_terms(raw_din, self._width_p)
        return (packed_din, raw_din)

import queue
import numpy as np
from typing import Optional, List

class ConvLayerModel():
    def __init__(self, dut=None, 
                 weights: Optional[List[List[List[List[int]]]]] = None, 
                 output_activation: Optional[List[List[List[int]]]] = None,
                 input_activation:  Optional[List[List[List[int]]]] = None,
                 **kwargs): # <-- All hardware dimensions collapse into **kwargs

        self._dut = dut
        self._input_activation  = input_activation 
        self._output_activation = output_activation

        # 1. Parameter Extraction
        if dut is not None:
            self._data_o = dut.data_o
            self._data_i = dut.data_i
            
            self._kernel_width = int(dut.KernelWidth.value)
            self._input_width  = int(dut.LineWidthPx.value)
            self._input_height = int(dut.LineCountPx.value)
            self._InBits       = int(dut.InBits.value)
            self._OutBits      = int(dut.OutBits.value)
            self._InChannels   = int(dut.InChannels.value)
            self._OutChannels  = int(dut.OutChannels.value)
            self._Stride       = int(dut.Stride.value)
        else:
            # If a parameter is missing, kwargs will throw a KeyError automatically
            try:
                self._kernel_width = int(kwargs["KernelWidth"])
                self._input_width  = int(kwargs["LineWidthPx"])
                self._input_height = int(kwargs["LineCountPx"])
                self._InBits       = int(kwargs["InBits"])
                self._OutBits      = int(kwargs["OutBits"])
                self._InChannels   = int(kwargs["InChannels"])
                self._OutChannels  = int(kwargs["OutChannels"])
                self._Stride       = int(kwargs["Stride"])
            except KeyError as e:
                raise ValueError(f"Missing required parameter when dut is None: {e}")

        # 2. Safety check for Weights (prevents the NoneType error!)
        if weights is None:
            raise ValueError("Weights must be provided to ConvLayerModel")

        # 3. State Initialization
        self._q = queue.SimpleQueue()
        self._r = 0
        self._c = 0
        self._OW = (self._input_width - self._kernel_width) // self._Stride + 1
        self._OH = (self._input_height - self._kernel_width) // self._Stride + 1

        self._buf = [np.zeros((self._kernel_width, self._input_width))/0 for _ in range(self._InChannels)]
        self._deqs = 0
        self._enqs = 0

        self.k = np.array(weights, dtype=int)
        assert self.k.shape == (self._OutChannels, self._InChannels, self._kernel_width, self._kernel_width)

        # 4. Validity Map Generation
        invalid_region = self._kernel_width - 1
        S = int(self._Stride)
        span_w = (self._input_width  - 1) - (self._kernel_width - 1)
        span_h = (self._input_height - 1) - (self._kernel_width - 1)

        assert span_w >= 0 and span_h >= 0, "Kernel exceeds image bounds"
        assert (span_w % S) == 0 and (span_h % S) == 0, "Invalid configuration: Stride does not tile evenly"
        
        self._valid_cycles = np.ones((self._input_height, self._input_width), dtype=bool)
        self._valid_cycles[:invalid_region, :] = False
        self._valid_cycles[:, :invalid_region] = False
        
        if S > 1:
            for r in range(invalid_region, self._input_height):
                for c in range(invalid_region, self._input_width):
                    if ((r - invalid_region) % S) != 0 or ((c - invalid_region) % S) != 0:
                        self._valid_cycles[r, c] = False

    # Now let's scale this up a little bit
    # You can define functions to do the steps in convolution
    def update_window(self, buf, inp):
        temp = buf.flatten()

        # Now shift everything by 1
        temp = np.roll(temp, -1, axis=0)

        # Add the new input, replacing the input that was "kicked out"
        temp[-1] = inp

        # Now reshape it back into the original buffer
        temp = np.reshape(temp, buf.shape)
        buf = temp
        return buf

    def apply_kernel(self, bufs):
        result = np.zeros(self._OutChannels, dtype=int)
        windows = [b[:, -self._kernel_width:].astype(int, copy=False) for b in bufs]

        for oc in range(self._OutChannels):
            acc = 0
            for ic in range(self._InChannels):
                win = windows[ic]

                if self._InBits == 1:
                    # BNN Mode: 0 -> -1, 1 -> +1
                    win_enc = np.where(win == 1, 1, -1)
                else:
                    # Multi-bit Mode: Ensure Python treats these as signed
                    # If 'win' contains raw bits from the DUT, sign-extend them
                    win_enc = np.vectorize(sign_extend)(win, self._InBits)

                acc += int((self.k[oc, ic] * win_enc).sum())

            if self._OutBits == 1:
                result[oc] = 1 if acc > 0 else 0
            else:
                result[oc] = acc

        return result

    def step(self, raw_val, in_fire=True):
        """
        Pure Python step function. 
        Takes raw Python integers, runs the math, tracks pixel coordinates, 
        and returns the expected output. No Cocotb dependencies.
        """
        if not in_fire:
            return None

        # Input activation tracking moved here from consume()
        if self._input_activation is not None:
            y_in = self._enqs // self._input_width
            x_in = self._enqs % self._input_width
            for ic in range(self._InChannels):
                self._input_activation[ic][y_in][x_in] = raw_val[ic]

        # Window updating moved here from consume()
        for ic, inp in enumerate(raw_val):
            self._buf[ic] = self.update_window(self._buf[ic], inp)

        # Validity checking moved here from consume()
        idx = self._enqs
        x = idx % self._input_width
        y = idx // self._input_width
        self._enqs += 1

        if y >= self._input_height or not self._valid_cycles[y, x]:
            return None

        # Kernel application moved here from consume()
        result = self.apply_kernel(self._buf)

        # Pixel coordinate tracking (_c, _r) moved here from produce().
        # This ensures the model tracks its state even in software-only mode.
        if self._output_activation is not None:
            for ch in range(self._OutChannels):
                self._output_activation[ch][self._r][self._c] = result[ch]

        self._deqs += 1
        self._c += 1
        if self._c >= self._OW:
            self._c = 0
            self._r += 1
            if self._r >= self._OH:
                self._r = 0

        # Must return a tuple for your unified pipeline runner
        return tuple(result)

    def consume(self):
        """
        Hardware-to-software translator. 
        It unpacks the bits, passes them to step(), and returns the expected result.
        """
        if self._dut is not None:
            packed = int(self._dut.data_i.value.integer)
            raw_val = unpack_terms(packed, int(self._dut.InBits.value), self._InChannels)

            # Forward the unpacked Python ints to the pure math pipeline
            expected = self.step(raw_val, in_fire=True)

            # Wrap in a list so the model runner queue can handle bursts correctly
            return [expected] if expected is not None else None

    def produce(self, expected):
        """
        Verifier. 
        State tracking (_c, _r, _deqs) was removed so it doesn't break in software mode.
        """
        assert_resolvable(self._data_o)
        if self._dut is not None:
            w = int(self._dut.OutBits.value)
        packed = int(self._data_o.value.integer)

        # Because _c and _r are already advanced by step(), we calculate 
        # the *current* check coordinates mathematically for your print statement.
        check_idx = self._deqs - 1
        check_r = check_idx // self._OW
        check_c = check_idx % self._OW

        for ch in range(self._OutChannels):
            raw = (packed >> (ch * w)) & ((1 << w) - 1)
            if w == 1:
                got = raw
            else:
                got = sign_extend(raw, w)
            
            exp = int(expected[ch])

            print(f"Output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: expected {exp}, got {got} (raw=0x{raw:x})")

            # _output_activation tracking was removed from here (it is now in step())

            assert got == exp, (
                f"Mismatch at output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: "
                f"expected {exp}, got {got} (raw=0x{raw:x})"
            )
    