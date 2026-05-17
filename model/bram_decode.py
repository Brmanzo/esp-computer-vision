"""
bram_decode.py  —  Decode SB_RAM40_4K INIT fields from Yosys ice40.json or ice40.asc.

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
    Auto-detect and run both:
        python3 -m model.bram_decode                  (runs on ice40.json and/or ice40.asc if they exist)

    Verify Yosys JSON netlist:
        python3 -m model.bram_decode ice40.json       (loops all hex ROMs automatically)
        python3 -m model.bram_decode ice40.json <fragment>
        python3 -m model.bram_decode ice40.json <fragment> <hex_file>

    Verify post-PNR physical ASC bitstream:
        python3 -m model.bram_decode ice40.asc        (loops all hex ROMs automatically)
        python3 -m model.bram_decode ice40.asc <hex_file>
"""

from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path

DATAPATH = Path(__file__).resolve().parent / "data"
ROM_PATH = DATAPATH / "roms" / "hex"

# ---------------------------------------------------------------------------
# Core helpers
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


def get_bram_cell_for_hex(bram_cells: list[str], hex_name: str) -> str | None:
    """
    Map a given hex filename to its synthesized BRAM cell name based on layer keywords.
    """
    layer_part = None
    if "layer_1" in hex_name:
        layer_part = "conv_layer_inst_1"
    elif "layer_2" in hex_name:
        layer_part = "conv_layer_inst_2"
    elif "layer_3" in hex_name:
        layer_part = "classifier_layer_inst_3"
        
    if not layer_part:
        return None
        
    is_hi = "hi" in hex_name
    
    for cell in bram_cells:
        if layer_part in cell:
            if "weight_rom" in cell or "rom" in cell:
                if is_hi and "rom_hi" in cell:
                    return cell
                elif not is_hi and "rom_hi" not in cell:
                    return cell
    return None


# ---------------------------------------------------------------------------
# Verification Routines
# ---------------------------------------------------------------------------

def run_asc_verification(target_path: Path, hex_file_path: Path | None = None) -> bool:
    """
    Run post-route ASC BRAM verification.
    """
    if not target_path.exists():
        print(f"ERROR: Bitstream file '{target_path}' not found.")
        return False

    if hex_file_path:
        hex_files_to_verify = [hex_file_path]
    else:
        if not ROM_PATH.exists():
            print(f"ERROR: ROM directory '{ROM_PATH}' not found.")
            return False
        hex_files_to_verify = sorted([ROM_PATH / f for f in os.listdir(ROM_PATH) if f.startswith("layer_") and f.endswith(".hex")])

    print(f"Verifying weight hex files inside physical bitstream '{target_path.name}'...")
    all_ok = True
    for hex_path in hex_files_to_verify:
        if not hex_path.exists():
            print(f"ERROR: Hex file '{hex_path}' not found.")
            all_ok = False
            continue
        
        try:
            res = subprocess.run(
                ["icebram", "-v", str(hex_path), str(hex_path)],
                input=target_path.read_bytes(),
                capture_output=True,
                check=True
            )
            stdout = res.stdout.decode()
            stderr = res.stderr.decode()
            output = stdout + stderr

            if "Found and replaced" in output:
                for line in output.splitlines():
                    if "Found and replaced" in line:
                        msg = line.strip().replace("and replaced", "instances")
                        print(f"  \033[92m✓ {hex_path.name}:\033[0m {msg}")
                        break
            else:
                print(f"  \033[91m✗ {hex_path.name}:\033[0m No matching BRAM memory footprint found in ASC file.")
                all_ok = False
        except FileNotFoundError:
            print("ERROR: 'icebram' tool not found on this system. Please make sure fpga-icestorm is installed.")
            return False
        except subprocess.CalledProcessError as e:
            print(f"  \033[91m✗ {hex_path.name}:\033[0m icebram execution failed (exit code {e.returncode})")
            print(e.stderr.decode())
            all_ok = False

    return all_ok


def run_json_verification(json_path: Path, fragment: str | None = None, hex_path: Path | None = None) -> bool:
    """
    Run synthesized JSON BRAM verification.
    """
    if not json_path.exists():
        print(f"ERROR: JSON file '{json_path}' not found.")
        return False

    try:
        with open(json_path) as f:
            d = json.load(f)
        top = d["modules"]["top"]
        bram_cells = [name for name, cell in top["cells"].items() if cell["type"] == "SB_RAM40_4K"]
    except Exception as e:
        print(f"ERROR reading JSON netlist: {e}")
        return False

    # Determine verification target list: [(fragment, hex_path), ...]
    if fragment and hex_path:
        hex_files_to_verify = [(fragment, hex_path)]
    elif fragment:
        # Check if fragment is actually a hex file path
        if Path(fragment).exists() or (ROM_PATH / fragment).exists():
            hex_file = Path(fragment) if Path(fragment).exists() else ROM_PATH / fragment
            matched_cell = get_bram_cell_for_hex(bram_cells, hex_file.name)
            if not matched_cell:
                print(f"ERROR: Could not automatically map hex file '{hex_file.name}' to a synthesized BRAM cell.")
                return False
            hex_files_to_verify = [(matched_cell, hex_file)]
        else:
            # View-only mode (no comparison)
            hex_files_to_verify = [(fragment, None)]
    else:
        # Batch mode
        if not ROM_PATH.exists():
            print(f"ERROR: ROM directory '{ROM_PATH}' not found.")
            return False
        hex_files_to_verify = []
        for f in sorted(os.listdir(ROM_PATH)):
            if f.startswith("layer_") and f.endswith(".hex"):
                hex_file = ROM_PATH / f
                matched_cell = get_bram_cell_for_hex(bram_cells, f)
                if matched_cell:
                    hex_files_to_verify.append((matched_cell, hex_file))
                else:
                    print(f"\033[93m⚠ WARNING:\033[0m Skipping '{f}', no matching synthesized BRAM cell found.")

    print(f"Verifying synthesized BRAMs in netlist '{json_path.name}'...")
    all_ok = True
    for frag, hex_p in hex_files_to_verify:
        try:
            words = decode_bram_words(json_path, frag)
            cell_name_short = frag.split('.')[-1] if '.' in frag else frag
            layer_hint = ""
            for part in frag.split('.'):
                if "layer_inst" in part:
                    layer_hint = f" ({part})"
                    break
            
            print(f"\nDecoded BRAM '{cell_name_short}'{layer_hint}:")
            print("  " + "  ".join(f"{w:04x}" for w in words[:16]))
            if len(words) > 16:
                print(f"  ... ({len(words) - 16} more)")

            if hex_p:
                if not hex_p.exists():
                    print(f"  \033[91m✗ ERROR:\033[0m Hex file '{hex_p}' not found.")
                    all_ok = False
                    continue
                matches, total = compare_to_hex(json_path, frag, hex_p)
                status = "✓ SUCCESS" if matches == total else "✗ MISMATCH"
                color = "\033[92m" if matches == total else "\033[91m"
                print(f"  {color}{status}:\033[0m compared against '{hex_p.name}': {matches}/{total} match")
                if matches != total:
                    all_ok = False
        except Exception as e:
            print(f"  \033[91m✗ ERROR on '{frag}':\033[0m {e}")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # If help menu is requested explicitly
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print("BRAM Decode & Verification Utility")
        print("==================================")
        print("Usage:")
        print("  1. Batch verify default files in current directory (ice40.json / ice40.asc):")
        print("     python3 -m model.bram_decode")
        print("\n  2. Verify Yosys JSON netlist cells:")
        print("     python3 -m model.bram_decode <netlist.json>                   (loops all hex ROMs)")
        print("     python3 -m model.bram_decode <netlist.json> <cell_fragment>   (decodes one cell)")
        print("     python3 -m model.bram_decode <netlist.json> <fragment> <hex>  (compares one cell)")
        print("\n  3. Verify physical post-route ASC bitstreams:")
        print("     python3 -m model.bram_decode <bitstream.asc>                  (loops all hex ROMs)")
        print("     python3 -m model.bram_decode <bitstream.asc> <hex_file>       (compares one file)")
        print("\nExamples:")
        print("  python3 -m model.bram_decode")
        print("  python3 -m model.bram_decode ice40.json")
        print("  python3 -m model.bram_decode ice40.asc")
        sys.exit(0)

    # 1. No arguments provided: Auto-detect and run
    if len(sys.argv) < 2:
        default_json = Path("ice40.json")
        default_asc = Path("ice40.asc")
        
        run_json = default_json.exists()
        run_asc = default_asc.exists()
        
        if not run_json and not run_asc:
            print("\033[93m⚠ Automatic Verification:\033[0m Neither 'ice40.json' nor 'ice40.asc' found in the current directory.")
            print("Please run synthesis/P&R first or provide a file path.\n")
            # Print help menu
            subprocess.run([sys.executable, "-m", "model.bram_decode", "--help"])
            sys.exit(0)
            
        all_success = True
        if run_json:
            print("================================================================================")
            print(f"★ Auto-detect: Found '{default_json.name}'. Starting synthesized BRAM verification...")
            print("================================================================================")
            all_success &= run_json_verification(default_json)
            print()
            
        if run_asc:
            print("================================================================================")
            print(f"★ Auto-detect: Found '{default_asc.name}'. Starting physical bitstream verification...")
            print("================================================================================")
            all_success &= run_asc_verification(default_asc)
            print()
            
        sys.exit(0 if all_success else 1)

    # 2. Arguments provided: Manual run
    target_path = Path(sys.argv[1])

    if target_path.suffix == ".asc":
        # ASC mode
        hex_file_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
        success = run_asc_verification(target_path, hex_file_path)
        sys.exit(0 if success else 1)
    else:
        # JSON mode
        fragment = sys.argv[2] if len(sys.argv) >= 3 else None
        hex_path = Path(sys.argv[3]) if len(sys.argv) >= 4 else None
        success = run_json_verification(target_path, fragment, hex_path)
        sys.exit(0 if success else 1)
