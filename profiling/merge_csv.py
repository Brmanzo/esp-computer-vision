import pandas as pd

import argparse

ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--file-1",    required=True,  help="Filepath for first CSV (e.g. dsp_chars_6.csv)")
ap.add_argument("--file-2",    required=True,  help="Filepath for second CSV (e.g. dsp_chars_12.csv)")
ap.add_argument("--out",       default="dsp_chars_combined.csv", help="Output CSV filepath")
args = ap.parse_args()

a = pd.read_csv(args.file_1)
b = pd.read_csv(args.file_2)

assert set(a.columns) == set(b.columns), "CSV files must have the same columns"
combined = pd.concat([a, b]).drop_duplicates(
    subset=['InChannels', 'InBits', 'PoolMode']
).reset_index(drop=True)
print(f"Combined {len(a)} rows from {args.file_1} and {len(b)} rows from {args.file_2} into {len(combined)} unique rows.")
combined.to_csv(args.out, index=False)
