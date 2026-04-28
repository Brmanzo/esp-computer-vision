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

class ConvLayerModel():
    def __init__(self, dut=None, 
                 weights: Optional[List[List[List[List[int]]]]] = None, 
                 output_activation: Optional[List[List[List[int]]]] = None,
                 input_activation:  Optional[List[List[List[int]]]] = None,
                 **kwargs): 

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
            self._Padding      = int(dut.Padding.value) if hasattr(dut, 'Padding') else 0
        else:
            try:
                self._kernel_width = int(kwargs["KernelWidth"])
                self._input_width  = int(kwargs["LineWidthPx"])
                self._input_height = int(kwargs["LineCountPx"])
                self._InBits       = int(kwargs["InBits"])
                self._OutBits      = int(kwargs["OutBits"])
                self._InChannels   = int(kwargs["InChannels"])
                self._OutChannels  = int(kwargs["OutChannels"])
                self._Stride       = int(kwargs["Stride"])
                self._Padding      = int(kwargs.get("Padding", 0))
            except KeyError as e:
                raise ValueError(f"Missing required parameter when dut is None: {e}")

        if weights is None:
            raise ValueError("Weights must be provided to ConvLayerModel")

        # Dimensions scaled for padding
        self._padded_width  = self._input_width + 2 * self._Padding
        self._padded_height = self._input_height + 2 * self._Padding

        self._OW = (self._padded_width - self._kernel_width) // self._Stride + 1
        self._OH = (self._padded_height - self._kernel_width) // self._Stride + 1

        # State Initialization
        self._input_queue = queue.SimpleQueue() # Buffers real input while processing padding
        self._r = 0
        self._c = 0
        self._x = 0 # Tracks internal padded X
        self._y = 0 # Tracks internal padded Y
        
        # Buffer width scales with padded width!
        self._buf = [np.zeros((self._kernel_width, self._padded_width)) for _ in range(self._InChannels)]
        self._deqs = 0
        self._enqs = 0

        self.k = np.array(weights, dtype=int)
        assert self.k.shape == (self._OutChannels, self._InChannels, self._kernel_width, self._kernel_width)

        # 4. Validity Map Generation (Based on PADDED dimensions)
        invalid_region = self._kernel_width - 1
        S = int(self._Stride)
        span_w = (self._padded_width  - 1) - (self._kernel_width - 1)
        span_h = (self._padded_height - 1) - (self._kernel_width - 1)

        assert span_w >= 0 and span_h >= 0, "Kernel exceeds image bounds"
        assert (span_w % S) == 0 and (span_h % S) == 0, "Invalid configuration: Stride does not tile evenly"
        
        self._valid_cycles = np.ones((self._padded_height, self._padded_width), dtype=bool)
        self._valid_cycles[:invalid_region, :] = False
        self._valid_cycles[:, :invalid_region] = False
        
        if S > 1:
            for r in range(invalid_region, self._padded_height):
                for c in range(invalid_region, self._padded_width):
                    if ((r - invalid_region) % S) != 0 or ((c - invalid_region) % S) != 0:
                        self._valid_cycles[r, c] = False

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

        for oc in range(self._OutChannels):
            acc = 0
            for ic in range(self._InChannels):
                win = windows[ic]
                if self._InBits == 1:
                    win_enc = np.where(win == 1, 1, -1)
                else:
                    win_enc = np.vectorize(sign_extend)(win, self._InBits)
                acc += int((self.k[oc, ic] * win_enc).sum())

            if self._OutBits == 1:
                result[oc] = 1 if acc > 0 else 0
            else:
                result[oc] = acc

        return result

    def _process_internal_cycles(self) -> List[tuple]:
        """
        Emulates the hardware's internal decoupled pipeline. 
        Will generate padding zeros automatically until it runs out of real data 
        in the input_queue. Returns a list of generated outputs.
        """
        results = []
        while True:
            # Check padding zones based on coordinates
            is_pad_x = (self._x < self._Padding) or (self._x >= self._input_width + self._Padding)
            is_pad_y = (self._y < self._Padding) or (self._y >= self._input_height + self._Padding)
            is_pad = is_pad_x or is_pad_y

            if is_pad:
                val = [0] * self._InChannels # Insert zero for padding cycle
            else:
                # If it's a real coordinate, we need real data from the queue
                if self._input_queue.empty():
                    break # Wait for Cocotb to feed us another pixel!
                
                val = self._input_queue.get()
                
                # Track Input Activations
                if self._input_activation is not None:
                    y_in = self._y - self._Padding
                    x_in = self._x - self._Padding
                    for ic in range(self._InChannels):
                        self._input_activation[ic][y_in][x_in] = val[ic]

            # 1. Update Windows
            for ic, inp in enumerate(val):
                self._buf[ic] = self.update_window(self._buf[ic], inp)

            # 2. Check Validity and Apply Kernel
            if self._y >= (self._kernel_width - 1) and self._valid_cycles[self._y, self._x]:
                res = self.apply_kernel(self._buf)
                
                if self._output_activation is not None:
                    for ch in range(self._OutChannels):
                        self._output_activation[ch][self._r][self._c] = res[ch]

                self._deqs += 1
                self._c += 1
                if self._c >= self._OW:
                    self._c = 0
                    self._r += 1
                    if self._r >= self._OH:
                        self._r = 0
                
                results.append(tuple(res))

            # 3. Advance Padded Counters
            self._x += 1
            if self._x >= self._padded_width:
                self._x = 0
                self._y += 1
                if self._y >= self._padded_height:
                    self._y = 0
                    # End of Frame: Break to avoid infinitely processing 
                    # the NEXT frame's top-padding before it starts.
                    break 

        return results

    def step(self, raw_val, in_fire=True):
        """
        Pushes a real pixel into the queue and spins the internal cycle machine.
        Returns a list of expected outputs (can be empty or have multiple tuples).
        """
        if not in_fire:
            return None

        self._enqs += 1
        self._input_queue.put(raw_val)
        
        # Process the pipeline until we stall waiting for the next pixel
        results = self._process_internal_cycles()
        
        return results if len(results) > 0 else None

    def consume(self):
        """
        Hardware-to-software translator. 
        """
        if self._dut is not None:
            packed = int(self._dut.data_i.value.integer)
            raw_val = unpack_terms(packed, int(self._dut.InBits.value), self._InChannels)

            # Forward the unpacked Python ints to the pure math pipeline
            # Note: step() now returns a list of tuples or None.
            return self.step(raw_val, in_fire=True) 

    def produce(self, expected):
        """
        Verifier. 
        """
        assert_resolvable(self._data_o)
        if self._dut is not None:
            w = int(self._dut.OutBits.value)
        packed = int(self._data_o.value.integer)

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

            assert got == exp, (
                f"Mismatch at output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: "
                f"expected {exp}, got {got} (raw=0x{raw:x})"
            )