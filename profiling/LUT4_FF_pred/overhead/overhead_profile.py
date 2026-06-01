from profiling.LUT4_FF_pred.overhead.deframer.deframer_profile import predict, _ceil_ple

# Fixed overhead constants — measured via `make stat-modules MOD=uart_cnn`
# (SB_LUT4 / total SB_DFF* counts for each sub-module, ESP=1 context)
uart         = (186, 79)   # uart_rx + uart_tx + uart top; was (82,35) — badly underestimated
class_framer = ( 19,  4)
skid_buffer  = ( 61, 45)   # updated from 60 → 61 to match measurement

def predict_overhead(uw: int, pn: int, ple: int) -> tuple[int, int]:
    LUT4, FF = predict(uw, pn, ple)
    LUT4 += uart[0] + class_framer[0] + skid_buffer[0]
    FF += uart[1] + class_framer[1] + skid_buffer[1]
    return LUT4, FF

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Query deframer LUT4/FF from sweep lookup")
    ap.add_argument("--uw",  type=int, required=True, help="UnpackedWidth")
    ap.add_argument("--pn",  type=int, required=True, help="PackedNum")
    ap.add_argument("--ple", type=int, required=True, help="PacketLenElems")
    args = ap.parse_args()

    try:
        lut4, ff = predict_overhead(uw=args.uw, pn=args.pn, ple=args.ple)
        ple_key = _ceil_ple(args.ple)
        note = f" (rounded up from {args.ple})" if ple_key != args.ple else ""
        print(f"predict(uw={args.uw}, pn={args.pn}, ple={ple_key}{note})  =>  LUT4={lut4}  FF={ff}")
    except (AssertionError, KeyError, ValueError) as e:
        print(f"Error: {e}")