from profiling.overhead.deframer.deframer_profile import predict, _ceil_ple

# Fixed Overhead constants
uart         = (82,35)
class_framer = (19,4)
skid_buffer  = (60,45)

def predict_overhead(uw: int, pn: int, ple: int) -> tuple[int, int]:
    LC, FF = predict(uw, pn, ple)
    LC += uart[0] + class_framer[0] + skid_buffer[0]
    FF += uart[1] + class_framer[1] + skid_buffer[1]
    return LC, FF

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Query deframer LC/FF from sweep lookup")
    ap.add_argument("--uw",  type=int, required=True, help="UnpackedWidth")
    ap.add_argument("--pn",  type=int, required=True, help="PackedNum")
    ap.add_argument("--ple", type=int, required=True, help="PacketLenElems")
    args = ap.parse_args()

    try:
        lc, ff = predict_overhead(uw=args.uw, pn=args.pn, ple=args.ple)
        ple_key = _ceil_ple(args.ple)
        note = f" (rounded up from {args.ple})" if ple_key != args.ple else ""
        print(f"predict(uw={args.uw}, pn={args.pn}, ple={ple_key}{note})  =>  LC={lc}  FF={ff}")
    except (AssertionError, KeyError, ValueError) as e:
        print(f"Error: {e}")