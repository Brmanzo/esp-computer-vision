%% classifier.m — Fit LUT4 and FF models for classifier_layer per TermBits corner
%% LUT4 = A*IC*CC*WB + B*IC + C*log2(CC) + D
%% FF   = B*IC + D   (A=C=0: data shows no CC·WB or log2(CC) dependence)
%%
%% Cost structure (LUT4):
%%   A*IC*CC*WB  — linear_layer parallel MACs
%%   B*IC        — global_max storage (InChannels x TermBits FFs, captured in corner)
%%   C*log2(CC)  — comparator_tree depth (binary tree over ClassCount nodes)
%%   D           — base overhead
%%
%% Cost structure (FF):
%%   B*IC        — input staging registers (scales with TB via corner)
%%   D           — fixed control overhead

LC_CAP = 5280;

T = readtable('../profiles/sweep_classifier_dsp_0.csv');
T = T(T.LUT4 <= LC_CAP, :);   % exclude cap-exceeded points from fit

tb_vals = unique(T.TermBits)';

lut4_rows = {};
ff_rows   = {};

for tb = tb_vals
    mask = T.TermBits == tb;
    if sum(mask) < 4, continue; end

    ic_s   = T.InChannels(mask);
    cc_s   = T.ClassCount(mask);
    wb_s   = T.WeightBits(mask);
    lut4_s = T.LUT4(mask);
    ff_s   = T.FF(mask);

    X = [ic_s.*cc_s.*wb_s, ic_s, log2(double(cc_s)), ones(sum(mask),1)];

    %% LUT4 fit
    if rank(X) < size(X, 2)
        X_lc = [ic_s.*wb_s, log2(double(cc_s)), ones(sum(mask),1)];
        c_lc = [0; X_lc \ lut4_s];
        flag = '*';
    else
        X_lc = X;
        c_lc = X_lc \ lut4_s;
        flag = '';
    end
    r2_lc = r2score(lut4_s, X_lc * c_lc(end-size(X_lc,2)+1:end));
    fprintf('TB=%d%s  LUT4=%.3f·IC·CC·WB + %.3f·IC + %.3f·log2(CC) + %.3f  R²=%.3f  (N=%d)\n', ...
        tb, flag, c_lc(1), c_lc(2), c_lc(3), c_lc(4), r2_lc, sum(mask));
    lut4_rows{end+1} = {tb, 'LUT4', c_lc(1), c_lc(2), c_lc(3), c_lc(4), r2_lc};

    %% FF fit — simplified: FF = B*IC + D
    X_ff = [ic_s, ones(sum(mask),1)];
    c_ff = X_ff \ ff_s;
    r2_ff = r2score(ff_s, X_ff * c_ff);
    fprintf('TB=%d  FF  =%.3f·IC + %.3f  R²=%.3f  (N=%d)\n', ...
        tb, c_ff(1), c_ff(2), r2_ff, sum(mask));
    ff_rows{end+1} = {tb, 'FF', 0, c_ff(1), 0, c_ff(2), r2_ff};
end

%% Export to classifier_coeffs.csv
all_rows = [lut4_rows, ff_rows];
out_T = cell2table(vertcat(all_rows{:}), 'VariableNames', ...
    {'TermBits','Model','A','B','C','D','R2'});
writetable(out_T, '../profiles/classifier_coeffs.csv');
fprintf('Exported %d corners to classifier_coeffs.csv\n', height(out_T));

lut4_T = out_T(strcmp(out_T.Model, 'LUT4'), :);
ff_T   = out_T(strcmp(out_T.Model, 'FF'),   :);

%% Plot: LUT4 predicted vs actual per TermBits slice
figure(1); clf;
n_tb = numel(tb_vals);
cols = ceil(sqrt(n_tb));
rows_plot = ceil(n_tb / cols);
t = tiledlayout(rows_plot, cols, 'TileSpacing', 'compact');
title(t, 'LUT4: predicted vs actual per TermBits');

for k = 1:height(lut4_T)
    tb = lut4_T.TermBits(k);
    mask = T.TermBits == tb;
    ic_s = T.InChannels(mask); cc_s = T.ClassCount(mask);
    wb_s = T.WeightBits(mask); lut4_s = T.LUT4(mask);
    X    = [ic_s.*cc_s.*wb_s, ic_s, log2(double(cc_s)), ones(sum(mask),1)];
    pred = X * [lut4_T.A(k); lut4_T.B(k); lut4_T.C(k); lut4_T.D(k)];

    nexttile;
    scatter(lut4_s, pred, 18, wb_s, 'filled'); hold on;
    lim = [0, LC_CAP * 1.05];
    plot(lim, lim, 'r--');
    xlim(lim); ylim(lim);
    xlabel('Actual LUT4'); ylabel('Predicted LUT4');
    title(sprintf('TB=%d  R²=%.3f', tb, lut4_T.R2(k)));
    cb = colorbar; cb.Label.String = 'WeightBits';
end

%% Plot: FF predicted vs actual per TermBits slice
figure(2); clf;
t2 = tiledlayout(rows_plot, cols, 'TileSpacing', 'compact');
title(t2, 'FF: predicted vs actual per TermBits');

for k = 1:height(ff_T)
    tb = ff_T.TermBits(k);
    mask = T.TermBits == tb;
    ic_s = T.InChannels(mask);
    wb_s = T.WeightBits(mask); ff_s = T.FF(mask);
    X_ff = [ic_s, ones(sum(mask),1)];
    pred = X_ff * [ff_T.B(k); ff_T.D(k)];

    nexttile;
    scatter(ff_s, pred, 18, wb_s, 'filled'); hold on;
    lim = [0, max(ff_s) * 1.05];
    plot(lim, lim, 'r--');
    xlim(lim); ylim(lim);
    xlabel('Actual FF'); ylabel('Predicted FF');
    title(sprintf('TB=%d  R²=%.3f', tb, ff_T.R2(k)));
    cb = colorbar; cb.Label.String = 'WeightBits';
end

%% Helper
function s = r2score(y, yhat)
    ss_res = sum((y - yhat).^2);
    ss_tot = sum((y - mean(y)).^2);
    s = 1 - ss_res / ss_tot;
end
