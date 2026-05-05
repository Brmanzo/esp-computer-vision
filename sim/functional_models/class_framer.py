from util.utilities import assert_resolvable, sim_verbose

class ClassFramerModel():
    def __init__(self, dut):
        self._dut        = dut
        self._class_i    = dut.class_i
        self._uart_o     = dut.uart_o

        self._bus_bits  = int(dut.BusBits.value)
        self._wakeup_cmd = int(dut.WakeupCmd.value)
        self._tail0      = int(dut.TailByte0.value)
        self._tail1      = int(dut.TailByte1.value)

        self._deqs  = 0

    def consume(self):
        """
        Called by ModelRunner on an input handshake (valid_i && ready_o).
        Returns a list of all bytes expected to appear on the output as a result.
        """
        assert_resolvable(self._class_i)     
        val = int(self._class_i.value)
        
        # We expect the class byte followed by the two protocol tail bytes
        return [val, self._tail0, self._tail1]
    
    def produce(self, expected):
        """
        Called by ModelRunner on an output handshake (valid_o && ready_i).
        'expected' is provided by the Runner from the list returned by consume().
        """
        assert_resolvable(self._uart_o)
        got = int(self._uart_o.value)
        self._deqs += 1

        if sim_verbose():
            print(f"Output #{self._deqs}: Expected 0x{expected:02X}, Got 0x{got:02X}")
            
        assert got == expected, (
            f"Mismatch on output #{self._deqs}: expected 0x{expected:02X}, got 0x{got:02X}"
        )
