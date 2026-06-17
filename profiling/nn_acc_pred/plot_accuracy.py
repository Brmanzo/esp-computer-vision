import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import re

def main():
    features_path = 'profiling/nn_acc_pred/profiles/accuracy_features.csv'
    df = pd.read_csv(features_path)
    
    results_path = 'profiling/nn_acc_pred/profiles/results.txt'
    fps_list = []
    
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'aborted' in line:
                continue
            match = re.search(r'\[.*?(\d+\.\d+)\s*fps\]', line)
            if match:
                fps_list.append(float(match.group(1)))
            else:
                fps_list.append(np.nan)
                
    min_len = min(len(fps_list), len(df))
    fps_list = fps_list[:min_len]
    df = df.iloc[:min_len].copy()
    df['fps'] = fps_list
    
    # Calculate log accuracy so the plots match the linear/parabola shape exactly
    df['log_acc'] = np.log(df['float_acc'])
    
    # Fit regression to get coefficients for the trend lines
    X = np.column_stack((np.ones(len(df)), df['pct_ternary'], df['lc'], df['depth'], df['depth']**2))
    y = df['log_acc']
    c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    C0, C1, C2, C3, C4 = c
    
    # We hold the other variables at their mean when drawing the isolated regression lines
    mean_ternary = df['pct_ternary'].mean()
    mean_lc = df['lc'].mean()
    mean_depth = df['depth'].mean()
    
    out_dir = 'profiling/nn_acc_pred/plots'
    os.makedirs(out_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16, 'figure.dpi': 150})
    
    # UCSC Color Palette
    ucsc_blue = '#003C6C'
    ucsc_gold = '#FDB515'
    ucsc_cyan = '#00B5E2'
    
    # Create custom colormap for FPS
    ucsc_cmap = LinearSegmentedColormap.from_list('ucsc_cmap', [ucsc_blue, ucsc_cyan, ucsc_gold])
    
    # 1. Ternary vs Log Acc (Negative Line)
    plt.figure(figsize=(10, 7))
    scatter1 = plt.scatter(df['pct_ternary'] * 100, df['log_acc'], 
                           c=df['fps'], cmap=ucsc_cmap, alpha=0.8, edgecolors='w', linewidth=0.5, s=60)
    
    x_ternary = np.linspace(df['pct_ternary'].min(), df['pct_ternary'].max(), 100)
    y_ternary = C0 + C1 * x_ternary + C2 * mean_lc + C3 * mean_depth + C4 * (mean_depth**2)
    plt.plot(x_ternary * 100, y_ternary, color=ucsc_blue, linewidth=3, linestyle='--', label='Regression Fit (Negative Linear)')
    
    plt.colorbar(scatter1, label='Throughput (FPS)')
    plt.xlabel('Percentage of Ternary Layers (%)')
    plt.ylabel('Natural Log of Accuracy (ln(float_acc))')
    plt.title('Log Accuracy vs. Ternary Datapath Percentage')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'acc_vs_ternary.svg'), format='svg')
    plt.close()
    
    # 2. LC vs Log Acc (Positive Line)
    plt.figure(figsize=(10, 7))
    scatter2 = plt.scatter(df['lc'], df['log_acc'], 
                           c=df['fps'], cmap=ucsc_cmap, alpha=0.8, edgecolors='w', linewidth=0.5, s=60)
    
    x_lc = np.linspace(df['lc'].min(), df['lc'].max(), 100)
    y_lc = C0 + C1 * mean_ternary + C2 * x_lc + C3 * mean_depth + C4 * (mean_depth**2)
    plt.plot(x_lc, y_lc, color=ucsc_blue, linewidth=3, linestyle='--', label='Regression Fit (Positive Linear)')
    
    plt.colorbar(scatter2, label='Throughput (FPS)')
    plt.xlabel('Logic Cell Utilization (LCs)')
    plt.ylabel('Natural Log of Accuracy (ln(float_acc))')
    plt.title('Log Accuracy vs. Logic Cell Cost')
    plt.axvline(x=5280, color=ucsc_gold, linestyle='--', linewidth=2, alpha=0.9, label='Hardware LC Cap (5280)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'acc_vs_lc.svg'), format='svg')
    plt.close()
    
    # 3. Depth vs Log Acc (Inverted Parabola)
    plt.figure(figsize=(10, 7))
    jittered_depth = df['depth'] + np.random.uniform(-0.15, 0.15, size=len(df))
    scatter3 = plt.scatter(jittered_depth, df['log_acc'], 
                           c=df['fps'], cmap=ucsc_cmap, alpha=0.8, edgecolors='w', linewidth=0.5, s=60)
    
    x_depth = np.linspace(df['depth'].min(), df['depth'].max(), 100)
    y_depth = C0 + C1 * mean_ternary + C2 * mean_lc + C3 * x_depth + C4 * (x_depth**2)
    plt.plot(x_depth, y_depth, color=ucsc_blue, linewidth=3, linestyle='--', label='Regression Fit (Inverted Parabola)')
    
    plt.colorbar(scatter3, label='Throughput (FPS)')
    plt.xlabel('Network Depth (Layers)')
    plt.ylabel('Natural Log of Accuracy (ln(float_acc))')
    plt.title('Log Accuracy vs. Network Depth')
    plt.xticks(sorted(df['depth'].unique()))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'acc_vs_depth.svg'), format='svg')
    plt.close()

if __name__ == '__main__':
    main()
