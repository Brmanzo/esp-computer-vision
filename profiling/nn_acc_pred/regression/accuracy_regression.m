%% accuracy_regression.m — Predict model accuracy
%%
%% This script fits a multiple linear regression model:
%%   ln(float_acc) = C0 + C1 * pct_ternary + C2 * lc + C3 * depth + C4 * depth_squared
%%

csv_path = '../profiles/accuracy_features.csv';

%% ── Read data ────────────────────────────────────────────────────────────────
data = readtable(csv_path);

pct_ternary = data.pct_ternary;
lc = data.lc;
depth = data.depth;
depth_squared = depth .^ 2;
acc = log(data.float_acc);

%% ── Regression ───────────────────────────────────────────────────────────────
% Predictors: [1, pct_ternary, lc, depth, depth_squared]
X = [ones(length(acc), 1), pct_ternary, lc, depth, depth_squared];
y = acc;

% Fit linear model
[b, bint, r, rint, stats] = regress(y, X);

C0 = b(1);
C1 = b(2);
C2 = b(3);
C3 = b(4);
C4 = b(5);
R_squared = stats(1);
F_stat = stats(2);
p_value = stats(3);

fprintf('\n=== Multiple Linear Regression Results ===\n');
fprintf('Formula: ln(float_acc) = %.4f + %.4f * pct_ternary + %.4f * lc + %.4f * depth + %.4f * depth^2\n', C0, C1, C2, C3, C4);
fprintf('R-squared: %.4f\n', R_squared);
fprintf('F-statistic: %.2f\n', F_stat);
fprintf('p-value: %e\n\n', p_value);

% Detailed summary
mdl = fitlm([pct_ternary, lc, depth, depth_squared], y, 'PredictorVars', {'PctTernary', 'LC', 'Depth', 'DepthSquared'}, 'ResponseVar', 'LogAccuracy');
disp(mdl);

%% ── Visualization ────────────────────────────────────────────────────────────
figure;
scatter3(depth, pct_ternary, acc, 'filled');
xlabel('Depth');
ylabel('% Ternary Datapath');
zlabel('Accuracy (float_acc)');
title('Accuracy vs. Architecture');
grid on;
