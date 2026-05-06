import numpy as np
from   pathlib import Path
import queue
from   typing import List, Optional

from util.utilities  import assert_resolvable, sim_verbose
from util.bitwise    import sign_extend, pack_terms, unpack_terms
from util.gen_inputs import gen_input_channels

def output_width(width_in, weight_width, kernel_width, in_channels, bias_bits=8):
    return str(calc_acc_bits(kernel_width, width_in, weight_width, in_channels, bias_bits))

def calc_acc_bits(kernel_width, in_bits, weight_bits, in_channels, bias_bits):
    terms = int(kernel_width) * int(kernel_width) * int(in_channels)
    max_input = 1 if int(in_bits) <= 2 else (1 << (int(in_bits) - 1))
    max_weight = 1 if int(weight_bits) <= 2 else (1 << (int(weight_bits) - 1))
    worst_case_sum = terms * max_input * max_weight
    wc_bits = worst_case_sum.bit_length() + 1
    acc_bits = max(wc_bits, int(bias_bits)) + 1
    return acc_bits

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
                 biases: Optional[List[int]] = None,
                 **kwargs): 

        self._dut = dut
        self._input_activation  = input_activation 
        self._output_activation = output_activation

        # 1. Parameter Extraction
        if dut is not None:
            self._data_o = dut.data_o
            self._data_i = dut.data_i
            
            self._kernel_width = int(dut.KernelWidth.value)
            self._weight_width = int(dut.WeightBits.value)
            self._input_width  = int(dut.LineWidthPx.value)
            self._input_height = int(dut.LineCountPx.value)
            self._bias_bits    = int(dut.BiasBits.value)
            self._InBits       = int(dut.InBits.value)
            self._OutBits      = int(dut.OutBits.value)
            self._InChannels   = int(dut.InChannels.value)
            self._OutChannels  = int(dut.OutChannels.value)
            self._Stride       = int(dut.Stride.value)
            self._Padding      = int(dut.Padding.value) if hasattr(dut, 'Padding') else 0
        else:
            try:
                self._kernel_width = int(kwargs["KernelWidth"])
                self._weight_width = int(kwargs["WeightBits"])
                self._bias_bits    = int(kwargs["BiasBits"])
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

        self._AccBits = calc_acc_bits(self._kernel_width, self._InBits, self._weight_width, self._InChannels, self._bias_bits)

        if weights is None:
            raise ValueError("Weights must be provided to ConvLayerModel")
        if biases is None:
            biases = [0] * self._OutChannels
        self._biases = biases

        # Dimensions scaled for padding
        self._padded_width  = self._input_width + 2 * self._Padding
        self._padded_height = self._input_height + 2 * self._Padding

        self._OW = (self._padded_width - self._kernel_width) // self._Stride + 1
        self._OH = (self._padded_height - self._kernel_width) // self._Stride + 1

        # State Initialization
        self._input_queue: 'queue.SimpleQueue[List[int]]' = queue.SimpleQueue() # Buffers real input while processing padding
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
        S = self._Stride
        span_w = (self._padded_width  - 1) - (self._kernel_width - 1)
        span_h = (self._padded_height - 1) - (self._kernel_width - 1)

        assert span_w >= 0 and span_h >= 0, "Kernel exceeds image bounds"
        # Note: stride need not evenly divide span — floor((span / S) + 1) output positions are produced.
        
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

            # Calculate the sum including bias
            biased_acc = acc + self._biases[oc]

            if self._OutBits == 1:
                # FIX: Use biased_acc here, not acc!
                result[oc] = 1 if biased_acc > 0 else 0
            else:
                # MSB Selection / Bit-slicing matching hardware behavior
                # 1. Hardware performs accumulation in 32 bits (SB_MAC16)
                # 2. neuron_dsp takes the AccBits LSBs
                # 3. filter_dsp takes the OutBits MSBs of the AccBits LSBs
                
                acc_mask = (1 << self._AccBits) - 1
                usable_acc = biased_acc & acc_mask
                
                # Take the top OutBits of the AccBits range
                # We need to treat usable_acc as signed within the AccBits range
                usable_acc_signed = sign_extend(usable_acc, self._AccBits)
                result[oc] = sign_extend((usable_acc_signed >> (self._AccBits - self._OutBits)) & ((1 << self._OutBits) - 1), self._OutBits)

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
        packed = int(self._data_o.value.integer)

        check_idx = self._deqs - 1
        check_r = check_idx // self._OW
        check_c = check_idx % self._OW

        for ch in range(self._OutChannels):
            raw = (packed >> (ch * self._OutBits)) & ((1 << self._OutBits) - 1)
            if self._OutBits == 1:
                got = raw
            else:
                got = sign_extend(raw, self._OutBits)
            
            exp = int(expected[ch])

            if sim_verbose():   
                print(f"Output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: expected {exp}, got {got} (raw=0x{raw:x})")

            assert got == exp, (
                f"Mismatch at output #{check_idx} (r={check_r}, c={check_c}) ch{ch}: "
                f"expected {exp}, got {got} (raw=0x{raw:x})"
            )