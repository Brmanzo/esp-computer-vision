%% accuracy_regression.m — Predict model accuracy from depth and growth rate
%%
%% This script fits a multiple linear regression model:
%%   float_acc = C0 + C1 * depth + C2 * growth_rate
%%
%% The growth_rate is defined as the final layer channels / first layer channels.
%%
%% Sources:  ../nn/tasks/generate/accuracy_features.csv

csv_path = '../profiles/accuracy_features.csv';

%% ── Read data ────────────────────────────────────────────────────────────────
data = readtable(csv_path);

depth = data.depth;
growth_rate = data.growth_rate;
acc = data.float_acc;

%% ── Regression ───────────────────────────────────────────────────────────────
% Predictors: [1, depth, growth_rate]
X = [ones(length(acc), 1), depth, growth_rate];
y = acc;

% Fit linear model
[b, bint, r, rint, stats] = regress(y, X);

C0 = b(1);
C1 = b(2);
C2 = b(3);
R_squared = stats(1);
F_stat = stats(2);
p_value = stats(3);

fprintf('\n=== Multiple Linear Regression Results ===\n');
fprintf('Formula: float_acc = %.4f + %.4f * depth + %.4f * growth_rate\n', C0, C1, C2);
fprintf('R-squared: %.4f\n', R_squared);
fprintf('F-statistic: %.2f\n', F_stat);
fprintf('p-value: %e\n\n', p_value);

% Detailed summary
mdl = fitlm([depth, growth_rate], y, 'PredictorVars', {'Depth', 'GrowthRate'}, 'ResponseVar', 'Accuracy');
disp(mdl);

%% ── Visualization ────────────────────────────────────────────────────────────
figure;
scatter3(depth, growth_rate, acc, 'filled');
xlabel('Depth (Conv Layers)');
ylabel('Growth Rate (L_n/L_1)');
zlabel('Accuracy (float_acc)');
title('Accuracy vs. Architecture');
grid on;
