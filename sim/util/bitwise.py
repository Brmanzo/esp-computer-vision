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

def pack_channels(channel_data: list[list[int]], bits: int) -> int:
    """
    Unified 2D packer.
    Input: [[ch0_t0, ch0_t1], [ch1_t0, ch1_t1]]
    Logic: Flattens to [ch0_t0, ch0_t1, ch1_t0, ch1_t1] then packs LSB->MSB.
    """
    # Flattens nested list: [term for sublist in main_list for term in sublist]
    flat_terms = [term for channel in channel_data for term in channel]
    return pack_terms(flat_terms, bits)

def unpack_channels(packed: int, bits: int, in_channels: int, term_count: int) -> list[list[int]]:
    """
    Unified 2D unpacker.
    Logic: Unpacks a flat list then reshapes it into [OC][IC] or [CH][TERM] structure.
    """
    # 1. Get the flat list from the generic atomic unpacker
    flat_list = unpack_terms(packed, bits, in_channels * term_count)
    
    # 2. Reshape into 2D list using slicing
    return [flat_list[i * term_count : (i + 1) * term_count] for i in range(in_channels)]