
from collections import deque
from utilities import assert_resolvable

def safe_int_from_value(val, *, x_as=0):
    """
    Convert a cocotb BinaryValue/LogicArray to int even if it contains X/Z.
    x_as=0 -> treat X/Z as 0
    x_as=1 -> treat X/Z as 1
    """
    s = val.binstr.lower()  # e.g. '00x1'
    if 'x' in s or 'z' in s:
        repl = '1' if x_as else '0'
        s = s.replace('x', repl).replace('z', repl)
    return int(s, 2)

def unpack_data_o_buffer(BufferWidth, BufferRows, InputChannels, packed_o):
    mask = (1 << BufferWidth) - 1
    out = [[0]*BufferRows for _ in range(InputChannels)]
    for ch in range(InputChannels):
        for r in range(BufferRows):
            bitpos = (ch * BufferRows + r) * BufferWidth
            out[ch][r] = (packed_o >> bitpos) & mask
    return out

class MultiDelayBufferModel():
    def __init__(self, dut=None, Delay=None, BufferRows=None, InputChannels=None, BufferWidth=None):
        self._dut = dut
        
        # Pull parameters from DUT if provided, otherwise use explicit kwargs
        if dut is not None:
            self._data_i = dut.data_i
            self._data_o = dut.data_o
            self._Delay         = int(dut.Delay.value)
            self._BufferRows    = int(dut.BufferRows.value)
            self._InputChannels = int(dut.InputChannels.value)
            self._BufferWidth   = int(dut.BufferWidth.value)
        else:
            self._Delay         = Delay
            self._BufferRows    = BufferRows
            self._InputChannels = InputChannels
            self._BufferWidth   = BufferWidth
            if self._BufferWidth is None or self._Delay is None or self._BufferRows is None or self._InputChannels is None:
                raise ValueError("All parameters must be provided")

        self._BufferWidth = int(self._BufferWidth)
        self._mask = (1 << self._BufferWidth) - 1
        self._warmup = self._Delay * self._BufferRows + 1
        self._fires = 0

        zero_init = [0] * self._BufferRows
        self._ram = [
            deque([zero_init.copy() for _ in range(self._Delay)], maxlen=self._Delay)
            for _ in range(self._InputChannels)
        ]
        
        self._wr_pipe = [0 for _ in range(self._InputChannels)]
        self._wr_valid = False

        self._regs = [
            [0 for _ in range(self._BufferRows - 1)]
            for _ in range(self._InputChannels)
        ]

    def step(self, words, in_fire=True):
        """Pure Python mathematical step (No DUT dependencies)"""
        if not in_fire:
            return None

        self._fires += 1
        rd_now = None

        if self._wr_valid:
            rd_now = []
            for ch in range(self._InputChannels):
                old = self._ram[ch].popleft()
                rd_now.append(old.copy())

                new_word0 = self._wr_pipe[ch] & self._mask

                row_heads = [0] * self._BufferRows
                row_heads[0] = new_word0

                for r in range(1, self._BufferRows):
                    row_heads[r] = self._regs[ch][r - 1]

                for r in range(self._BufferRows - 1):
                    self._regs[ch][r] = old[r]

                self._ram[ch].append(row_heads)
        else:
            rd_now = [self._ram[ch][0].copy() for ch in range(self._InputChannels)]

        self._wr_pipe = [int(w) & self._mask for w in words]
        self._wr_valid = True

        out = None
        latency_threshold = (self._Delay * self._BufferRows) + 1
        if self._fires >= latency_threshold:
            out = rd_now

        # Unified runner requires tuples
        return tuple(out) if out is not None else None

    def consume(self):
        """Cocotb interface that calls the pure Python step"""
        packed_in = int(self._data_i.value.integer)
        words = [
            (packed_in >> (ch * self._BufferWidth)) & self._mask 
            for ch in range(self._InputChannels)
        ]
        return self.step(words, in_fire=True)

    def produce(self, expected):
        """Cocotb verification interface"""
        val = self._data_o.value
        if not val.is_resolvable:
            raise AssertionError(
                f"data_o contains X/Z on output handshake at fires={self._fires}. "
                f"data_o.binstr={val.binstr}"
            )

        got_packed = int(val.integer)
        got = unpack_data_o_buffer(
            self._BufferWidth,
            self._BufferRows,
            self._InputChannels,
            got_packed
        )

        expected_list = list(expected)
        assert got == expected_list, f"Mismatch: got={got} expected={expected_list}"