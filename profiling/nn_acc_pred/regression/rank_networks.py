import argparse
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Rank networks by predicted accuracy using Model 1.")
    parser.add_argument("-n", "--top", type=int, default=10, help="Number of top networks to display")
    args = parser.parse_args()

    csv_path = Path(__file__).parent.parent / "profiles" / "accuracy_features.csv"
    if not csv_path.exists():
        print("Error: CSV not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    X = df[["pct_ternary", "lc"]].copy()
    X = sm.add_constant(X)
    y = df["float_acc"]
    
    # Train final model on ALL data for best predictions
    model = sm.OLS(y, X).fit()
    
    # Predict accuracy for all networks
    df["predicted_acc"] = model.predict(X)
    
    # Sort by predicted accuracy descending
    ranked_df = df.sort_values(by="predicted_acc", ascending=False).reset_index(drop=True)
    
    print(f"=== Top {args.top} Networks by Predicted Accuracy ===")
    print(f"{'Rank':<5} | {'Idx':<4} | {'Predicted Acc':<13} | {'Actual Acc':<10} | {'% Ternary':<9} | {'LC Util'}")
    print("-" * 70)
    
    for i in range(min(args.top, len(ranked_df))):
        row = ranked_df.iloc[i]
        print(f"{i+1:<5} | {int(row['idx']):<4} | {row['predicted_acc']:.4f}        | {row['float_acc']:.4f}     | {row['pct_ternary']:.1%}     | {int(row['lc'])}")

if __name__ == "__main__":
    main()
