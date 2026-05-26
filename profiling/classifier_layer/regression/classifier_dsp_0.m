%% classifier.m — Fit LC model for classifier_layer per TermBits corner
%% LC = A*IC*CC*WB + B*IC + C*log2(CC) + D
%%
%% Cost structure:
%%   A*IC*CC*WB  — linear_layer parallel MACs (scales with all three)
%%   B*IC        — global_max storage (InChannels x TermBits FFs, captured in corner)
%%   C*log2(CC)  — comparator_tree depth (binary tree over ClassCount nodes)
%%   D           — base overhead

LC_CAP = 5280;

T = readtable('../profiles/sweep_classifier.csv');
T = T(T.LC <= LC_CAP, :);   % exclude cap-exceeded points from fit

tb_vals = unique(T.TermBits)';

rows = {};
for tb = tb_vals
    mask = T.TermBits == tb;
    if sum(mask) < 4, continue; end

    ic_s = T.InChannels(mask);
    cc_s = T.ClassCount(mask);
    wb_s = T.WeightBits(mask);
    lc_s = T.LC(mask);

    X = [ic_s.*cc_s.*wb_s, ic_s, log2(double(cc_s)), ones(sum(mask),1)];

    if rank(X) < size(X, 2)
        % Rank-deficient: drop IC*CC*WB interaction and fit simpler model
        X = [ic_s.*wb_s, log2(double(cc_s)), ones(sum(mask),1)];
        c_full = [0; X \ lc_s];
        flag = '*';
    else
        c_full = X \ lc_s;
        flag = '';
    end

    r2 = 1 - sum((lc_s - X*c_full(end-size(X,2)+1:end)).^2) / ...
             sum((lc_s - mean(lc_s)).^2);
    fprintf('TB=%d%s  LC=%.3f·IC·CC·WB + %.3f·IC + %.3f·log2(CC) + %.3f  R²=%.3f  (N=%d)\n', ...
        tb, flag, c_full(1), c_full(2), c_full(3), c_full(4), r2, sum(mask));
    rows{end+1} = {tb, 'LC', c_full(1), c_full(2), c_full(3), c_full(4), r2};
end

%% Export to classifier_coeffs.csv
out_T = cell2table(vertcat(rows{:}), 'VariableNames', ...
    {'TermBits','Model','A','B','C','D','R2'});
writetable(out_T, '../profiles/classifier_coeffs.csv');
fprintf('Exported %d corners to classifier_coeffs.csv\n', height(out_T));

%% Plot: predicted vs actual per TermBits slice
figure(1); clf;
n_tb = numel(tb_vals);
cols = ceil(sqrt(n_tb));
rows_plot = ceil(n_tb / cols);
t = tiledlayout(rows_plot, cols, 'TileSpacing', 'compact');
title(t, 'LC: predicted vs actual per TermBits');

for k = 1:height(out_T)
    tb = out_T.TermBits(k);
    mask = T.TermBits == tb;
    ic_s = T.InChannels(mask); cc_s = T.ClassCount(mask);
    wb_s = T.WeightBits(mask); lc_s = T.LC(mask);
    X    = [ic_s.*cc_s.*wb_s, ic_s, log2(double(cc_s)), ones(sum(mask),1)];
    pred = X * [out_T.A(k); out_T.B(k); out_T.C(k); out_T.D(k)];

    nexttile;
    scatter(lc_s, pred, 18, wb_s, 'filled'); hold on;
    lim = [0, LC_CAP * 1.05];
    plot(lim, lim, 'r--');
    xlim(lim); ylim(lim);
    xlabel('Actual LC'); ylabel('Predicted LC');
    title(sprintf('TB=%d  R²=%.3f', tb, out_T.R2(k)));
    cb = colorbar; cb.Label.String = 'WeightBits';
end
