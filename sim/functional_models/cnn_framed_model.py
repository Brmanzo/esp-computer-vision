# cnn_framed_model.py
# Integrated functional model for cnn_framed DUT (deframer + CNN + class_framer).

from __future__ import annotations
from typing import List, Optional


class FramedPictureGenerator:
    """
    Generates the byte stream that the Python demo sends:
      [0xA5, 0x5A]  — header
      [packed_byte_0, ..., packed_byte_N-1]  — pixels packed LSB-first (8 px / byte)
    """

    def __init__(self, pixels: List[int], bus_bits: int = 8):
        n_bytes = (len(pixels) + bus_bits - 1) // bus_bits
        image_bytes: List[int] = []
        for i in range(n_bytes):
            byte = 0
            for bit in range(bus_bits):
                px_idx = i * bus_bits + bit
                if px_idx < len(pixels) and pixels[px_idx]:
                    byte |= (1 << bit)
            image_bytes.append(byte)
        self._sequence = [0xA5, 0x5A] + image_bytes
        self._ptr = 0

    def generate(self) -> tuple[int, List[int]]:
        byte = self._sequence[self._ptr] if self._ptr < len(self._sequence) else 0
        self._ptr += 1
        return (byte, [byte])

    def total_bytes(self) -> int:
        return len(self._sequence)


class CnnFramedModel:
    """
    Combined deframer + CNN + class_framer functional model for the cnn_framed DUT.

    Implements the ModelRunner consume/produce interface operating on 8-bit bytes.

    consume() — called on each input byte handshake (valid_i & ready_o).
      Runs a deframer state machine: skips header bytes, unpacks 8 pixels per
      image byte and feeds them through cnn_model.step(). When the last image
      byte produces a CNN result it returns [class_id, tail0, tail1].

    produce() — called on each output byte handshake (valid_o & ready_i).
      The class_framer always outputs a 0x99 wakeup byte before any class
      result. Because the wakeup fires before any consume() result is queued,
      the ModelRunner sees an empty queue and skips it automatically. Subsequent
      bytes are verified in order: class_id → tail0 → tail1.

    results — list of class IDs captured from produce() calls.
    """

    HEADER0 = 0
    HEADER1 = 1
    FORWARD = 2

    def __init__(
        self,
        dut,
        cnn_model,
        header0: int = 0xA5,
        header1: int = 0x5A,
        tail0: int = 0xA5,
        tail1: int = 0x5A,
        packed_num: int = 8,
        packet_len_bytes: int = 9600,
    ):
        self._dut = dut
        self._cnn = cnn_model
        self._header0 = header0
        self._header1 = header1
        self._tail0 = tail0
        self._tail1 = tail1
        self._packed_num = packed_num
        self._packet_len = packet_len_bytes

        self._state = self.HEADER0
        self._remaining = 0

        self.results: List[int] = []
        self._deqs = 0
        # Position within each 3-byte output group (0=class, 1=tail0, 2=tail1)
        self._out_pos = 0

    def consume(self):
        from util.utilities import assert_resolvable
        assert_resolvable(self._dut.data_i)
        byte = int(self._dut.data_i.value) & 0xFF

        if self._state == self.HEADER0:
            if byte == self._header0:
                self._state = self.HEADER1
            return None

        elif self._state == self.HEADER1:
            if byte == self._header1:
                self._state = self.FORWARD
                self._remaining = self._packet_len
            elif byte == self._header0:
                pass  # stay in HEADER1
            else:
                self._state = self.HEADER0
            return None

        else:  # FORWARD
            result = None
            for bit_pos in range(self._packed_num):
                pixel = (byte >> bit_pos) & 1
                r = self._cnn.step([pixel])
                if r is not None:
                    result = r

            self._remaining -= 1
            if self._remaining <= 0:
                self._state = self.HEADER0

            if result is not None:
                class_id = result[0]
                return [class_id, self._tail0, self._tail1]
            return None

    def produce(self, expected: int):
        from util.utilities import assert_resolvable, sim_verbose
        assert_resolvable(self._dut.data_o)
        got = int(self._dut.data_o.value) & 0xFF
        self._deqs += 1

        if sim_verbose():
            print(f"Framed output #{self._deqs}: expected 0x{expected:02X}, got 0x{got:02X}")

        assert got == expected, (
            f"Framed output mismatch #{self._deqs}: expected 0x{expected:02X}, got 0x{got:02X}"
        )

        # Record the class byte (first of each [class, tail0, tail1] triplet)
        if self._out_pos == 0:
            self.results.append(got)
        self._out_pos = (self._out_pos + 1) % 3
