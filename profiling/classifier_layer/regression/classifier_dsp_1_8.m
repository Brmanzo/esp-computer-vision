%% classifier_dsp_1_8.m — Regress LC savings for classifier_layer DSPCount > 0.
%% Fits LC = A*IC*CC + B*IC + C*log2(CC) + D per (TermBits, DSPCount) corner,
%% then appends those corners to classifier_coeffs.csv.
DSP_CAP = 8;

T = readtable('../profiles/sweep_classifier_dsp_1_8.csv');

% Only DSP > 0 rows have a meaningful savings value
T = T(T.DSPCount > 0, :);
T.savings         = T.LC_dsp0 - T.LC;
T.savings_per_dsp = T.savings ./ T.DSPCount;

% Group by (TB, DSPCount) only — WB does not affect LC in DSP mode
% (weights are in ROM; SB_MAC16 width is fixed hardware)
tb_vals  = unique(T.TermBits)';
dsp_vals = unique(T.DSPCount)';

dsp_rows = cell(numel(tb_vals) * numel(dsp_vals), 1);
row_idx  = 0;
for tb = tb_vals
  for d = dsp_vals
    mask = T.TermBits==tb & T.DSPCount==d;
    if sum(mask) < 4, continue; end
    ic_s = T.InChannels(mask); cc_s = T.ClassCount(mask); lc_s = T.LC(mask);
    wb_s = T.WeightBits(mask);
    X = [ic_s.*cc_s.*wb_s, ic_s, log2(cc_s), ones(sum(mask),1)];

    if rank(X) < size(X, 2)
        % Rank-deficient: drop IC·CC interaction term
        X = [ic_s, log2(cc_s), ones(sum(mask),1)];
        c_full = [0; X \ lc_s];
        flag = '*';
    else
        c_full = X \ lc_s;
        flag = '';
    end

    r2 = 1 - sum((lc_s - X*c_full(end-size(X,2)+1:end)).^2) / ...
             sum((lc_s - mean(lc_s)).^2);
    fprintf('TB=%d DSP=%d%s  LC=%.2f·IC·CC + %.2f·IC + %.2f·log2(CC) + %.2f  R²=%.3f  (N=%d)\n', ...
        tb, d, flag, c_full(1), c_full(2), c_full(3), c_full(4), r2, sum(mask));
    row_idx = row_idx + 1;
    dsp_rows{row_idx} = {tb, d, 'LC', c_full(1), c_full(2), c_full(3), c_full(4), r2};
  end
end

dsp_rows = dsp_rows(1:row_idx);

%% Append DSP>0 corners to classifier_coeffs.csv
dsp_T = cell2table(vertcat(dsp_rows{:}), 'VariableNames', ...
    {'TermBits','DSPCount','Model','A','B','C','D','R2'});

coeffs_path = '../profiles/classifier_coeffs.csv';
if isfile(coeffs_path)
    existing = readtable(coeffs_path);
    % Back-fill DSPCount=0 for legacy rows written before this column existed
    if ~ismember('DSPCount', existing.Properties.VariableNames)
        existing.DSPCount = zeros(height(existing), 1);
    end
    % Drop stale DSP>0 rows so re-runs don't accumulate duplicates
    dsp0_only = existing(existing.DSPCount == 0, :);
    combined  = [dsp0_only; dsp_T];
else
    combined = dsp_T;
end
writetable(combined, coeffs_path);
fprintf('Written %d DSP>0 corners; total %d rows in classifier_coeffs.csv\n', ...
    height(dsp_T), height(combined));

%% Plot 1: LC vs ClassCount/DSPCount (aggregated trend)
figure(1); clf;
T.cc_per_dsp = T.ClassCount ./ T.DSPCount;

scatter(T.cc_per_dsp, T.LC, 20, T.InChannels, 'filled');
xlabel('ClassCount / DSPCount');
ylabel('LC');
title('LC vs class outputs per DSP block');
cb = colorbar; cb.Label.String = 'InChannels';

%% Plot 2: FF vs ClassCount/DSPCount
figure(2); clf;
scatter(T.cc_per_dsp, T.FF, 20, T.InChannels, 'filled');
xlabel('ClassCount / DSPCount');
ylabel('FF');
title('FF vs class outputs per DSP block');
cb = colorbar; cb.Label.String = 'InChannels';
ylim([0, 5280]);

%% Plot 3: Fractional LC vs CC/DSP ratio, one line per DSPCount (TB=4 slice)
figure(3); clf; hold on;
T2 = T(T.TermBits == 4, :);
T2.lc_frac = T2.LC ./ T2.LC_dsp0;   % 1.0 = no savings, lower = more savings

colors = lines(DSP_CAP);
for d = 1:DSP_CAP
    mask = T2.DSPCount == d;
    if sum(mask) < 2, continue; end
    [ratio, idx] = sort(T2.ClassCount(mask) ./ d);
    frac = T2.lc_frac(mask); frac = frac(idx);
    plot(ratio, frac, '-o', 'Color', colors(d,:), 'DisplayName', sprintf('DSP=%d', d));
end

xlabel('ClassCount / DSPCount');
ylabel('LC / LC_{DSP=0}  (1.0 = no savings)');
title('Fractional LC vs CC/DSP ratio  (TB=4, all IC)');
yline(1.0, 'k--');
legend('Location', 'best');

%% Summary
fprintf('Global mean savings/DSP : %.2f LC\n', mean(T.savings_per_dsp));
fprintf('Mean R² of DSP>0 LC regression corners: %.3f\n', mean(dsp_T.R2));
fprintf('Std deviation           : %.2f LC  (%.1f%%)\n', ...
    std(T.savings_per_dsp), 100*std(T.savings_per_dsp)/mean(T.savings_per_dsp));
fprintf('Min / Max               : %.1f / %.1f LC\n', ...
    min(T.savings_per_dsp), max(T.savings_per_dsp));
