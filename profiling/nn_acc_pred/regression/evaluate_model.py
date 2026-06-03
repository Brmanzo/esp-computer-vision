import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

def main():
    csv_path = Path(__file__).parent.parent / "profiles" / "accuracy_features.csv"
    if not csv_path.exists():
        print("Error: CSV not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df[["pct_ternary", "lc"]].copy()
    X = sm.add_constant(X)
    y = df["float_acc"]
    
    # Split the data into 80% training and 20% testing manually
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Fit the model ONLY on the training data
    model = sm.OLS(y_train, X_train).fit()
    
    print("=== Model Training Results (80% of data) ===")
    print(f"Formula: float_acc = {model.params['const']:.4f} + {model.params['pct_ternary']:.4f}*pct_ternary + {model.params['lc']:.6f}*lc")
    print(f"Training R-squared: {model.rsquared:.4f}")
    
    # Predict on the unseen testing data
    y_pred = model.predict(X_test)
    
    # Calculate error metrics
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    
    print("\n=== Model Testing Results (20% unseen data) ===")
    print(f"Mean Absolute Error (MAE):       {mae:.4f}  (Model predictions are off by ~{mae*100:.1f}% on average)")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.4f}")
    
    # Show a few random examples of True vs Predicted
    print("\n--- Example Predictions on Unseen Networks ---")
    results = pd.DataFrame({"True_Acc": y_test, "Pred_Acc": y_pred})
    print(results.sample(5, random_state=42).to_string())

if __name__ == "__main__":
    main()
