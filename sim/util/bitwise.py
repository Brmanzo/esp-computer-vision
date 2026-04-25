# Utility functions for bit manipulation, packing/unpacking, and sign extension.

def sign_extend(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    sign_bit = 1 << (width - 1)
    return (value ^ sign_bit) - sign_bit

def pack_terms(terms, in_bits):
    packed = 0
    mask = (1 << in_bits) - 1
    for i, x in enumerate(terms):
        packed |= (x & mask) << (i * in_bits)
    return packed

def unpack_terms(packed, term_bits, input_count):
    unpacked = []
    mask = (1 << term_bits) - 1
    for i in range(input_count):
        raw = (packed >> (i * term_bits)) & mask

        if term_bits == 1:
            unpacked.append(raw)
        else:
            unpacked.append(sign_extend(raw, term_bits))
            
    return unpacked