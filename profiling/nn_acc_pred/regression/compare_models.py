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
    y = df["float_acc"]
    n = len(y)
    
    # ---------------------------------------------------------
    # Model 1: Linear
    # float_acc = C0 + C1*pct_ternary + C2*lc
    # ---------------------------------------------------------
    X1 = df[["pct_ternary", "lc"]].copy()
    X1 = sm.add_constant(X1)
    mod1 = sm.OLS(y, X1).fit()
    
    y_pred1 = mod1.predict(X1)
    ss_res1 = np.sum((y - y_pred1)**2)
    k1 = X1.shape[1]
    
    # ---------------------------------------------------------
    # Model 2: Log / Bell-Curve
    # ln(float_acc) = C0 + C1*pct_ternary + C2*lc + C3*depth + C4*depth_squared
    # ---------------------------------------------------------
    X2 = df[["pct_ternary", "lc", "depth"]].copy()
    X2["depth_squared"] = X2["depth"] ** 2
    X2 = sm.add_constant(X2)
    y_log = np.log(y)
    
    mod2 = sm.OLS(y_log, X2).fit()
    
    # Back-transform predictions to linear space
    y_pred_log = mod2.predict(X2)
    y_pred2 = np.exp(y_pred_log)
    ss_res2 = np.sum((y - y_pred2)**2)
    k2 = X2.shape[1]
    
    # ---------------------------------------------------------
    # Calculate Comparable Metrics in the Linear Domain
    # ---------------------------------------------------------
    ss_tot = np.sum((y - np.mean(y))**2)
    
    r2_1 = 1 - (ss_res1 / ss_tot)
    r2_2 = 1 - (ss_res2 / ss_tot)
    
    # Standard AIC/BIC formulas for least squares (omitting the constant log(2*pi) term 
    # which cancels out when comparing models)
    aic1 = n * np.log(ss_res1 / n) + 2 * k1
    aic2 = n * np.log(ss_res2 / n) + 2 * k2
    
    bic1 = n * np.log(ss_res1 / n) + k1 * np.log(n)
    bic2 = n * np.log(ss_res2 / n) + k2 * np.log(n)
    
    print("=== Model 1 (Linear): float_acc ~ pct_ternary + lc ===")
    print(f"Comparable R-squared: {r2_1:.4f}")
    print(f"Equivalent AIC:       {aic1:.2f}")
    print(f"Equivalent BIC:       {bic1:.2f}")
    print("")
    print("=== Model 2 (Log/Bell-Curve): ln(float_acc) ~ pct_ternary + lc + depth + depth^2 ===")
    print(f"Comparable R-squared: {r2_2:.4f}  (calculated after e^x transformation)")
    print(f"Equivalent AIC:       {aic2:.2f}")
    print(f"Equivalent BIC:       {bic2:.2f}")
    print("")
    print("--- Conclusion ---")
    if aic1 < aic2:
        print("Model 1 has a LOWER AIC. It is the better model (less overfit).")
    else:
        print("Model 2 has a LOWER AIC. The added depth complexity is mathematically justified!")

if __name__ == "__main__":
    main()
