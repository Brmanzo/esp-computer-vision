import pandas as pd
import statsmodels.api as sm

def main():
    try:
        df = pd.read_csv("profiling/nn_acc_pred/profiles/accuracy_features.csv")
    except FileNotFoundError:
        print("Error: Could not find accuracy_features.csv. Did you run the export script?")
        return

    import numpy as np
    
    X = df[["pct_ternary", "lc", "depth"]].copy()
    X["depth_squared"] = X["depth"] ** 2
    X = X[["pct_ternary", "lc", "depth", "depth_squared"]]
    X = sm.add_constant(X)
    y = np.log(df["float_acc"])

    model = sm.OLS(y, X).fit()

    print("=== Python Multiple Linear Regression Results ===")
    print("Formula: ln(float_acc) = C0 + C1 * pct_ternary + C2 * lc + C3 * depth + C4 * depth_squared")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"F-statistic: {model.fvalue:.2f}")
    print(f"p-value (F-stat): {model.f_pvalue:e}")
    print("\nCoefficients:")
    print(model.params.to_string())
    print("\nP-values:")
    print(model.pvalues.to_string())

if __name__ == "__main__":
    main()
