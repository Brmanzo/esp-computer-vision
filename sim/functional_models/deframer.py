
from util.utilities import assert_resolvable, sim_verbose

class DeframerModel:
    HEADER0 = 0
    HEADER1 = 1
    FORWARD = 2

    def __init__(self, dut):
        self._dut = dut
        self._unpacked_o = dut.unpacked_o
        self._data_i = dut.data_i

        self._UnpackedWidth  = int(dut.UnpackedWidth.value)
        self._PackedNum      = int(dut.PackedNum.value)
        self._PackedWidth    = int(dut.PackedWidth.value)
        self._PacketLenElems = int(dut.PacketLenElems.value)
        self._HeaderByte0    = int(dut.HeaderByte0.value)
        self._HeaderByte1    = int(dut.HeaderByte1.value)

        self._mask = (1 << self._UnpackedWidth) - 1

        self._state = self.HEADER0
        self._in_remaining = 0
        
        self._deqs = 0
        self._enqs = 0
    
    def consume(self):
        assert_resolvable(self._data_i)
        b = int(self._data_i.value) & ((1 << (self._PackedWidth)) - 1)
        
        # Detecting first byte
        if self._state == self.HEADER0:
            if b == self._HeaderByte0:
                self._state = self.HEADER1
            return None
        
        # Detecting second byte after the first byte
        elif self._state == self.HEADER1:
            if b == self._HeaderByte1:
                self._state = self.FORWARD
                self._in_remaining = self._PacketLenElems * self._PackedNum
            elif b == self._HeaderByte0:
                # Stay in HEADER1 if we see another HEADER0
                self._state = self.HEADER1
            else:
                # Reset if unexpected byte
                self._state = self.HEADER0
            return None
        # Once both bytes detected, enqueue the rest of the packet
        elif self._state == self.FORWARD:
            if self._in_remaining > 0:
                expected_outputs = []
                for step in range(self._PackedNum):
                    val = ((b >> (self._UnpackedWidth * step)) & self._mask)
                    expected_outputs.append(val)

                self._in_remaining -= self._PackedNum
                self._enqs += 1

                if self._in_remaining <= 0:
                    self._state = self.HEADER0
                return expected_outputs
            else:
                # Packet complete: ignore until next header
                self._state = self.HEADER0
                return False

    def produce(self, expected):
        assert_resolvable(self._unpacked_o)
        got = int(self._unpacked_o.value) & self._mask
       
        self._deqs += 1
        if sim_verbose():
            print(f'Output #{self._deqs}: Got unpacked: {got}, Expected unpacked: {expected}')

        assert got == expected, (
            f"Mismatch on output #{self._deqs}: expected {expected}, got {got}"
        )