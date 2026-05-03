# pool_layer.py
from   pathlib import Path
import numpy as np
from   typing import List, Optional

from util.utilities import assert_resolvable
from util.bitwise   import sign_extend, unpack_terms, pack_terms
from util.gen_inputs import gen_input_channels

class RandomDataGenerator:
    def __init__(self, dut):
        self._width_p = int(dut.InBits.value)
        self._InChannels = int(dut.InChannels.value)

    def generate(self):
        raw_din = gen_input_channels(self._width_p, self._InChannels)
        packed_din = pack_terms(raw_din, self._width_p)
        return (packed_din, raw_din)
class PoolLayerModel():
    def __init__(self, dut=None,
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
            self._PoolMode     = int(dut.PoolMode.value)
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
                self._PoolMode     = int(kwargs["PoolMode"])
            except KeyError as e:
                raise ValueError(f"Missing required parameter when dut is None: {e}")

        self._r = 0
        self._c = 0
        self._OW = (self._input_width - self._kernel_width) // self._Stride + 1
        self._OH = (self._input_height - self._kernel_width) // self._Stride + 1

        self._buf = [np.zeros((self._kernel_width,self._input_width), dtype=np.int64) for _ in range(self._InChannels)]
        self._deqs = 0
        self._enqs = 0

        self._in_idx = 0

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


    # ==========================================
    # UNTOUCHED MATH HELPER FUNCTIONS
    # ==========================================
    def update_window(self, buf, inp):
        temp = buf.flatten()
        temp = np.roll(temp, -1, axis=0)
        temp[-1] = inp
        temp = np.reshape(temp, buf.shape)
        buf = temp
        return buf

    def apply_kernel(self, bufs):
        result = np.zeros(self._OutChannels, dtype=int)
        windows = [b[:, -self._kernel_width:].astype(int, copy=False) for b in bufs]

        for ch in range(self._OutChannels):
            if self._PoolMode == 0:
                result[ch] = np.max(windows[ch])
            else:
                if self._OutBits == 1:
                    acc = np.sum(windows[ch] == 1) - np.sum(windows[ch] == 0)
                    shift = int(np.log2(windows[ch].size))
                    result[ch] = (acc >> shift) & 1
                else:
                    result[ch] = np.sum(windows[ch]) // windows[ch].size

        return result

    # ==========================================
    # REFACTORED PIPELINE INTERFACES
    # ==========================================
    def step(self, raw_val, in_fire=True):
        if not in_fire:
            return None

        # Input activation tracking
        if self._input_activation is not None:
            y_in = self._enqs // self._input_width
            x_in = self._enqs % self._input_width
            for ic in range(self._InChannels):
                self._input_activation[ic][y_in][x_in] = raw_val[ic]

        # Advance windows
        for ic, inp in enumerate(raw_val):
            self._buf[ic] = self.update_window(self._buf[ic], inp)

        # Check validity
        idx = self._enqs
        x = idx % int(self._input_width)
        y = idx // int(self._input_width)
        self._enqs += 1

        if y >= int(self._input_height) or not self._valid_cycles[y, x]:
            return None

        # Compute expected
        result = self.apply_kernel(self._buf)

        # Output activation and coordinate tracking moved here from produce()
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

        return tuple(result)

    def consume(self):
        if self._dut is not None:
            assert_resolvable(self._data_i)
            packed = int(self._data_i.value.integer)
            raw_val = unpack_terms(packed, int(self._dut.InBits.value), self._InChannels)

            # Call the pure math pipeline
            expected = self.step(raw_val, in_fire=True)
            return [expected] if expected is not None else None

    def produce(self, expected):
        if self._dut is not None:
            assert_resolvable(self._data_o)
            w = int(self._dut.OutBits.value)
            packed = int(self._data_o.value.integer)

            # Calculate the *current* coordinates for printing
            check_idx = self._deqs - 1
            check_r = check_idx // self._OW
            check_c = check_idx % self._OW

            for ch in range(self._OutChannels):
                got_raw = (packed >> (ch * w)) & ((1 << w) - 1)
                
                if w == 1:
                    got = got_raw
                else:
                    got = sign_extend(got_raw, w)
                    
                exp = int(expected[ch])

                print(f"Output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: expected {exp}, got {got}")

                assert got == exp, (
                    f"Mismatch at output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: expected {exp}, got {got} "
                )