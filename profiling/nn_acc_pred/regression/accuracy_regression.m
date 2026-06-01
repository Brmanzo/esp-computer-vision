%% accuracy_regression.m — Predict model accuracy
%%
%% This script fits a multiple linear regression model:
%%   float_acc = C0 + C1 * channel_bits + C2 * pct_ternary
%%

csv_path = '../profiles/accuracy_features.csv';

%% ── Read data ────────────────────────────────────────────────────────────────
data = readtable(csv_path);

channel_bits = data.channel_bits;
pct_ternary = data.pct_ternary;
acc = data.float_acc;

%% ── Regression ───────────────────────────────────────────────────────────────
% Predictors: [1, channel_bits, pct_ternary]
X = [ones(length(acc), 1), channel_bits, pct_ternary];
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
fprintf('Formula: float_acc = %.4f + %.4f * channel_bits + %.4f * pct_ternary\n', C0, C1, C2);
fprintf('R-squared: %.4f\n', R_squared);
fprintf('F-statistic: %.2f\n', F_stat);
fprintf('p-value: %e\n\n', p_value);

% Detailed summary
mdl = fitlm([channel_bits, pct_ternary], y, 'PredictorVars', {'ChannelBits', 'PctTernary'}, 'ResponseVar', 'Accuracy');
disp(mdl);

%% ── Visualization ────────────────────────────────────────────────────────────
figure;
scatter3(channel_bits, pct_ternary, acc, 'filled');
xlabel('Channel * Input Bits');
ylabel('% Ternary Datapath');
zlabel('Accuracy (float_acc)');
title('Accuracy vs. Architecture');
grid on;
