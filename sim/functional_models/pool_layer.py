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
    def __init__(self, dut, 
                 output_activation: Optional[List[List[List[int]]]] = None,
                 input_activation: Optional[List[List[List[int]]]] = None):
       
        self._kernel_width = int(dut.KernelWidth.value)
        self._f = np.ones((self._kernel_width,self._kernel_width), dtype=int)
        self._dut = dut
        self._data_o = dut.data_o
        self._data_i = dut.data_i

        self._input_width = int(dut.LineWidthPx.value)
        self._input_height = int(dut.LineCountPx.value)
        self._OutBits = int(dut.OutBits.value)
        self._InChannels  = int(dut.InChannels.value)
        self._OutChannels = int(dut.OutChannels.value)
        self._Stride = int(dut.Stride.value)
        self._PoolMode = int(dut.PoolMode.value)

        self._output_activation = output_activation
        self._input_activation = input_activation
        self._r = 0
        self._c = 0
        self._OW = (self._input_width - self._kernel_width) // self._Stride + 1
        self._OH = (self._input_height - self._kernel_width) // self._Stride + 1

        # We're going to initialize _buf with NaN so that we can
        # detect when the output should be not an X in simulation
        # Buffer for all input channels, storing the most recent kernel_width values for each channel
        self._buf = [np.zeros((self._kernel_width,self._input_width))/0 for _ in range(self._InChannels)]
        self._deqs = 0
        self._enqs = 0

        # kernel 4D array storing all kernels in each filter: [OC][IC][K][K]
        self._in_idx = 0

        # Valid kernel positions within input image depending on stride and kernel size
        invalid_region = self._kernel_width - 1
        S = int(self._Stride)
        span_w = (self._input_width  - 1) - (self._kernel_width - 1)
        span_h = (self._input_height - 1) - (self._kernel_width - 1)

        assert span_w >= 0 and span_h >= 0, "Kernel exceeds image bounds"
        assert (span_w % S) == 0 and (span_h % S) == 0, "Invalid configuration: Stride does not tile evenly"
        
        self._valid_cycles = np.ones((self._input_height, self._input_width), dtype=bool)
        self._valid_cycles[:invalid_region, :] = False
        self._valid_cycles[:, :invalid_region] = False
        
        # If stride larger than normal, invalidate the skipped over elements
        if S > 1:
            for r in range(invalid_region, self._input_height):
                for c in range(invalid_region, self._input_width):
                    if ((r - invalid_region) % S) != 0 or ((c - invalid_region) % S) != 0:
                        self._valid_cycles[r, c] = False

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

        for ch in range(self._OutChannels):
            if self._PoolMode == 0:
                result[ch] = np.max(windows[ch])
            else:
                if self._OutBits == 1:
                    # acc += 1 for 1, acc -= 1 for 0
                    acc = np.sum(windows[ch] == 1) - np.sum(windows[ch] == 0)
                    shift = int(np.log2(windows[ch].size))
                    # Arithmetic shift and mask the LSB
                    result[ch] = (acc >> shift) & 1
                else:
                    result[ch] = np.sum(windows[ch]) // windows[ch].size

        return result

    def consume(self):
        assert_resolvable(self._data_i)
        packed = int(self._data_i.value.integer)
        packed_data_i = unpack_terms(packed, int(self._dut.InBits.value), self._InChannels)

        # 1. Record input activation if requested
        if self._input_activation is not None:
            y_in = self._enqs // self._input_width
            x_in = self._enqs % self._input_width
            for ic in range(self._InChannels):
                self._input_activation[ic][y_in][x_in] = packed_data_i[ic]

        # advance windows on EVERY accepted input
        for ic, inp in enumerate(packed_data_i):
            self._buf[ic] = self.update_window(self._buf[ic], inp)

        idx = self._enqs
        x = idx % int(self._input_width)
        y = idx // int(self._input_width)
        self._enqs += 1

        if y >= int(self._input_height):
            return None

        if not self._valid_cycles[y, x]:
            return None

        # compute expected NOW, while _buf matches this accepted input position
        expected = self.apply_kernel(self._buf)
        return expected

    def produce(self, expected):
        assert_resolvable(self._data_o)

        w = int(self._dut.OutBits.value)
        packed = int(self._data_o.value.integer)

        for ch in range(self._OutChannels):
            got_raw = (packed >> (ch * w)) & ((1 << w) - 1)
            
            if w == 1:
                got = got_raw
            else:
                got = sign_extend(got_raw, w)
                
            exp = int(expected[ch])

            print(f"Output #{self._deqs} (r={self._r}, c={self._c}) ch{ch}: expected {exp}, got {got}")

            if self._output_activation is not None:
                self._output_activation[ch][self._r][self._c] = got

            assert got == exp, (
                f"Mismatch at output #{self._deqs} (r={self._r}, c={self._c}) ch{ch}: expected {exp}, got {got} "
            )

        # Advance pixel position ONCE per output handshake/vector
        self._c += 1
        if self._c >= self._OW:
            self._c = 0
            self._r += 1
            if self._r >= self._OH:
                self._r = 0