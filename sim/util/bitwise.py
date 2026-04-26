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

def unpack_terms(packed, term_bits, input_count, signed=True):
    unpacked = []
    mask = (1 << term_bits) - 1
    for i in range(input_count):
        raw = (packed >> (i * term_bits)) & mask
        
        # Logic merge: 
        # If unsigned OR it's a single bit, take raw. 
        # Otherwise, sign_extend.
        if not signed or term_bits == 1:
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

def unpack_channels(packed: int, bits: int, in_channels: int, term_count: int, signed: bool = True) -> list[list[int]]:
    """
    Unified 2D unpacker supporting both signed and unsigned data.
    """
    # 1. Use the atomic unpacker to get a flat list
    # We pass the signed flag down or handle it in the terms loop
    flat_list = unpack_terms(packed, bits, in_channels * term_count, signed=signed)
    
    # 2. Reshape into [in_channels][term_count]
    return [flat_list[i * term_count : (i + 1) * term_count] for i in range(in_channels)]

def unpack_weights(packed_val: int, WW: int, OC: int, IC: int) -> list[list[int]]:
    """
    Reconstructs [OC][IC] weights using the unified 2D unpacker.
    """
    return unpack_channels(
        packed=packed_val, 
        bits=WW, 
        in_channels=OC, 
        term_count=IC
    )

def unpack_biases(packed_val: int, BW: int, OC: int) -> list[int]:
    """
    Reconstructs 1D bias list using the atomic unpacker.
    """
    return unpack_terms(
        packed=packed_val, 
        term_bits=BW, 
        input_count=OC
    )

def unpack_kernel_weights(packed_val: int, WW: int, OC: int, IC: int, K: int) -> list[list[list[list[int]]]]:
    """
    Reconstructs the 4D [OC][IC][K][K] weights matrix.
    Logic: Unpacks a flat list of all terms, then reshapes.
    """
    # 1. Extract all individual signed weights into a flat 1D list
    total_elements = OC * IC * K * K
    flat_list = unpack_terms(packed_val, WW, total_elements)
    
    # 2. Reshape the flat list into [OC][IC][K][K]
    # We work backwards from the innermost dimension (K)
    return [
        [
            [
                flat_list[oc*IC*K*K + ic*K*K + r*K : oc*IC*K*K + ic*K*K + (r+1)*K]
                for r in range(K)
            ]
            for ic in range(IC)
        ]
        for oc in range(OC)
    ]