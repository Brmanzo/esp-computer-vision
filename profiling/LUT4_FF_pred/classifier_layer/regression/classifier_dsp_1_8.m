%% classifier_dsp_1_8.m — Regress LUT4 savings for classifier_layer DSPCount > 0.
%% Fits LUT4 = A*IC*CC + B*IC + C*log2(CC) + D per (TermBits, DSPCount) corner,
%% then appends those corners to classifier_coeffs.csv.
DSP_CAP = 8;

T = readtable('../profiles/sweep_classifier_dsp_1_8.csv');

% Only DSP > 0 rows have a meaningful savings value
T = T(T.DSPCount > 0, :);
T.savings         = T.LUT4_dsp0 - T.LUT4;
T.savings_per_dsp = T.savings ./ T.DSPCount;

% Group by (TB, DSPCount) only — WB does not affect LUT4 in DSP mode
% (weights are in ROM; SB_MAC16 width is fixed hardware)
tb_vals  = unique(T.TermBits)';
dsp_vals = unique(T.DSPCount)';

lut4_rows = cell(numel(tb_vals) * numel(dsp_vals), 1);
ff_rows   = cell(numel(tb_vals), 1);
lut4_idx  = 0;
ff_idx    = 0;

%% LUT4 fit per (TB, DSP)
for tb = tb_vals
  for d = dsp_vals
    mask = T.TermBits==tb & T.DSPCount==d;
    if sum(mask) < 4, continue; end
    ic_s   = T.InChannels(mask); cc_s = T.ClassCount(mask);
    lut4_s = T.LUT4(mask);
    wb_s   = T.WeightBits(mask);
    X = [ic_s.*cc_s.*wb_s, ic_s, log2(cc_s), ones(sum(mask),1)];

    if rank(X) < size(X, 2)
        X = [ic_s, log2(cc_s), ones(sum(mask),1)];
        c_full = [0; X \ lut4_s];
        flag = '*';
    else
        c_full = X \ lut4_s;
        flag = '';
    end
    r2 = r2score(lut4_s, X * c_full(end-size(X,2)+1:end));
    fprintf('TB=%d DSP=%d%s  LUT4=%.2f·IC·CC + %.2f·IC + %.2f·log2(CC) + %.2f  R²=%.3f  (N=%d)\n', ...
        tb, d, flag, c_full(1), c_full(2), c_full(3), c_full(4), r2, sum(mask));
    lut4_idx = lut4_idx + 1;
    lut4_rows{lut4_idx} = {tb, d, 'LUT4', c_full(1), c_full(2), c_full(3), c_full(4), r2};
  end
end

%% FF fit per TB — pools all DSP>0 rows; FF = A*IC + B*CC + C*DSP + D
%% DSPCount=-1 in CSV signals "applies to all DSP>0" for this TB corner
for tb = tb_vals
    mask  = T.TermBits == tb;
    if sum(mask) < 4, continue; end
    ic_s  = T.InChannels(mask);
    cc_s  = T.ClassCount(mask);
    dsp_s = T.DSPCount(mask);
    ff_s  = T.FF(mask);

    X_ff  = [ic_s, cc_s, dsp_s, ones(sum(mask),1)];
    c_ff  = X_ff \ ff_s;
    r2_ff = r2score(ff_s, X_ff * c_ff);
    fprintf('TB=%d  FF  =%.2f·IC + %.2f·CC + %.2f·DSP + %.2f  R²=%.3f  (N=%d)\n', ...
        tb, c_ff(1), c_ff(2), c_ff(3), c_ff(4), r2_ff, sum(mask));
    ff_idx = ff_idx + 1;
    ff_rows{ff_idx} = {tb, -1, 'FF', c_ff(1), c_ff(2), c_ff(3), c_ff(4), r2_ff};
end

lut4_rows = lut4_rows(1:lut4_idx);
ff_rows   = ff_rows(1:ff_idx);

%% Append DSP>0 corners to classifier_coeffs.csv
all_dsp = [lut4_rows; ff_rows];
dsp_T = cell2table(vertcat(all_dsp{:}), 'VariableNames', ...
    {'TermBits','DSPCount','Model','A','B','C','D','R2'});

coeffs_path = '../profiles/classifier_coeffs.csv';
if isfile(coeffs_path)
    existing = readtable(coeffs_path);
    % Back-fill DSPCount=0 for legacy rows written before this column existed
    if ~ismember('DSPCount', existing.Properties.VariableNames)
        existing.DSPCount = zeros(height(existing), 1);
    end
    % Drop stale DSP>0 and FF sentinel rows so re-runs don't accumulate duplicates
    dsp0_only = existing(existing.DSPCount == 0, :);
    combined  = [dsp0_only; dsp_T];
else
    combined = dsp_T;
end
writetable(combined, coeffs_path);
fprintf('Written %d DSP>0 corners; total %d rows in classifier_coeffs.csv\n', ...
    height(dsp_T), height(combined));

%% Plot 1: LUT4 vs ClassCount/DSPCount (aggregated trend)
figure(1); clf;
T.cc_per_dsp = T.ClassCount ./ T.DSPCount;

scatter(T.cc_per_dsp, T.LUT4, 20, T.InChannels, 'filled');
xlabel('ClassCount / DSPCount');
ylabel('LUT4');
title('LUT4 vs class outputs per DSP block');
cb = colorbar; cb.Label.String = 'InChannels';

%% Plot 2: FF vs ClassCount/DSPCount
figure(2); clf;
scatter(T.cc_per_dsp, T.FF, 20, T.InChannels, 'filled');
xlabel('ClassCount / DSPCount');
ylabel('FF');
title('FF vs class outputs per DSP block');
cb = colorbar; cb.Label.String = 'InChannels';
ylim([0, 5280]);

%% Plot 3: Fractional LUT4 vs CC/DSP ratio, one line per DSPCount (TB=4 slice)
figure(3); clf; hold on;
T2 = T(T.TermBits == 4, :);
T2.lut4_frac = T2.LUT4 ./ T2.LUT4_dsp0;   % 1.0 = no savings, lower = more savings

colors = lines(DSP_CAP);
for d = 1:DSP_CAP
    mask = T2.DSPCount == d;
    if sum(mask) < 2, continue; end
    [ratio, idx] = sort(T2.ClassCount(mask) ./ d);
    frac = T2.lut4_frac(mask); frac = frac(idx);
    plot(ratio, frac, '-o', 'Color', colors(d,:), 'DisplayName', sprintf('DSP=%d', d));
end

xlabel('ClassCount / DSPCount');
ylabel('LUT4 / LC_{DSP=0}  (1.0 = no savings)');
title('Fractional LUT4 vs CC/DSP ratio  (TB=4, all IC)');
yline(1.0, 'k--');
legend('Location', 'best');

%% Summary
lut4_dsp_T = dsp_T(strcmp(dsp_T.Model, 'LUT4'), :);
fprintf('Global mean savings/DSP : %.2f LUT4\n', mean(T.savings_per_dsp));
fprintf('Mean R² of DSP>0 LUT4 regression corners: %.3f\n', mean(lut4_dsp_T.R2));
fprintf('Std deviation           : %.2f LUT4  (%.1f%%)\n', ...
    std(T.savings_per_dsp), 100*std(T.savings_per_dsp)/mean(T.savings_per_dsp));
fprintf('Min / Max               : %.1f / %.1f LUT4\n', ...
    min(T.savings_per_dsp), max(T.savings_per_dsp));

%% Helper
function s = r2score(y, yhat)
    ss_res = sum((y - yhat).^2);
    ss_tot = sum((y - mean(y)).^2);
    s = 1 - ss_res / ss_tot;
end
