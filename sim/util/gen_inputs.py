
import random
from   util.bitwise import pack_terms

def gen_random_unsigned(width, rng):
    """Generates a single unsigned integer."""
    max_val = (1 << width) - 1
    return rng.randint(0, max_val)

def gen_random_signed(width, rng):
    """Generates a single signed (or BNN binary) integer."""
    if width == 1:
        return rng.randint(0, 1)
    max_val = (1 << (width - 1)) - 1
    min_val = -(1 << (width - 1))
    return rng.randint(min_val, max_val)

def gen_input_channels(InBits: int, IC: int, seed: int | None = None):
    rng = random.Random(seed)
    return [gen_random_signed(InBits, rng) for _ in range(IC)]

def gen_kernels(WW: int, OC: int, IC: int, K: int, seed: int | None = None):
    rng = random.Random(seed)
    # Generate the 4D values
    vals = [gen_random_signed(WW, rng) for _ in range(OC * IC * K * K)]
    return pack_terms(vals, WW)

def gen_weights(WW: int, OC: int, IC: int= 1, seed: int | None = None):
    rng = random.Random(seed)
    # Generate the 2D values
    vals = [gen_random_signed(WW, rng) for _ in range(OC * IC)]
    return pack_terms(vals, WW)

def gen_biases(BW: int, OC: int = 1, seed: int | None = None):
    rng = random.Random(seed)
    vals = [gen_random_signed(BW, rng) for _ in range(OC)]
    return pack_terms(vals, BW)