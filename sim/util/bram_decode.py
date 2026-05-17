"""
bram_decode.py  —  Decode SB_RAM40_4K INIT fields from Yosys ice40.json.

Background
----------
Yosys maps logic [Width-1:0] rom [Depth-1:0] to SB_RAM40_4K via
brams_map.v.  The INIT_0..INIT_F parameters in the JSON are NOT a
straightforward dump of the hex file — two layers of scrambling apply:

1.  INIT bit ordering  (from brams_map.v  slice_init function)
        for i in 0..255:
            ri = i
            a  = {idx, ri[7:4], ri[0], ri[1], ri[2], ri[3]}   # 12 bits
            INIT_k[i] = MEMORY_FLAT[a]

    Where MEMORY_FLAT is the logical memory laid out as
        MEMORY_FLAT[word * WIDTH + bit] = memory[word][bit]

    This means:
        word = (k << 4) | (i >> 4)          — upper 8 bits of a
        bit  = bit_reverse4(i & 0xF)        — lower 4 bits of a, reversed

2.  JSON string convention  (Yosys MSB-first)
        INIT_k_string[j] = INIT_k bit (255 - j)
    so  INIT_k[i]        = INIT_k_string[255 - i]

Usage
-----
    from sim.util.bram_decode import decode_bram_words

    words = decode_bram_words(ice40_json_path, cell_name_fragment)
    # words is a list of integers, one per memory address (0-indexed)

    or run directly:
        python3 sim/util/bram_decode.py [ice40.json] [cell_fragment] [hex_file]
"""

from __future__ import annotations
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def _bit_reverse4(x: int) -> int:
    """Reverse the 4 bits of x (0-15)."""
    return ((x & 1) << 3) | ((x & 2) << 1) | ((x & 4) >> 1) | ((x & 8) >> 3)


def decode_bram_words(
    ice40_json: str | Path,
    cell_name_fragment: str,
    width: int = 16,
    depth: int = 256,
) -> list[int]:
    """
    Return a list of *depth* integers decoded from the first SB_RAM40_4K
    cell whose name contains *cell_name_fragment*.

    Parameters
    ----------
    ice40_json          : path to ice40.json (Yosys synthesis output)
    cell_name_fragment  : substring to match against cell names
    width               : data width in bits (default 16)
    depth               : address depth (default 256; must be <= 256 for one BRAM)
    """
    with open(ice40_json) as f:
        d = json.load(f)
    top = d["modules"]["top"]

    init_strings: dict[int, str] = {}
    for name, cell in top["cells"].items():
        if cell["type"] != "SB_RAM40_4K":
            continue
        if cell_name_fragment not in name:
            continue
        params = cell["parameters"]
        for i in range(16):
            k = f"INIT_{i:X}"
            val = params.get(k, "0" * 256).replace("x", "0").replace("X", "0")
            init_strings[i] = val
        break
    else:
        raise ValueError(f"No SB_RAM40_4K cell matching '{cell_name_fragment}' found")

    # memory[addr][bit] decoded from INIT strings
    memory: list[list[int | None]] = [[None] * width for _ in range(depth)]

    for k, s in init_strings.items():
        for i in range(256):
            word = (k << 4) | (i >> 4)
            bit  = _bit_reverse4(i & 0xF)
            if word < depth and bit < width:
                memory[word][bit] = int(s[255 - i])

    # Assemble bit-lists into integers
    result = []
    for addr in range(depth):
        bits = memory[addr]
        val  = sum(b << pos for pos, b in enumerate(bits) if b is not None)
        result.append(val)

    return result


# ---------------------------------------------------------------------------
# Convenience: compare decoded words against a hex file
# ---------------------------------------------------------------------------

def compare_to_hex(
    ice40_json: str | Path,
    cell_name_fragment: str,
    hex_file: str | Path,
    width: int = 16,
    depth: int = 256,
) -> tuple[int, int]:
    """
    Decode BRAM words and compare against *hex_file* ($readmemh format).

    Returns (num_matches, num_compared).
    """
    words = decode_bram_words(ice40_json, cell_name_fragment, width, depth)

    with open(hex_file) as f:
        expected = [int(line.strip(), 16) for line in f if line.strip()]

    n = min(len(words), len(expected))
    matches = sum(1 for a, b in zip(words[:n], expected[:n]) if a == b)
    return matches, n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ice40.json")
    fragment  = sys.argv[2] if len(sys.argv) > 2 else "weight_rom_inst.rom.0.0"
    hex_path  = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    words = decode_bram_words(json_path, fragment)
    print(f"Decoded {len(words)} words from cell matching '{fragment}':")
    print("  " + "  ".join(f"{w:04x}" for w in words[:16]))
    if len(words) > 16:
        print(f"  ... ({len(words) - 16} more)")

    if hex_path:
        matches, total = compare_to_hex(json_path, fragment, hex_path)
        print(f"\nCompared against {hex_path}: {matches}/{total} match")
