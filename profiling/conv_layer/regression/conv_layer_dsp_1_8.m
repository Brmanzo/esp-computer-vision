%% conv_layer_dsp_1-8.m — From a representative sampling of valid outch/dsp assignments
%% Estimate savings over the combinational dsp=0, and fit a regression model to extrapolate beyond the measured points.
DSP_CAP = 8;

T = readtable('../profiles/sweep_conv_dsp_1-8.csv');

% Only DSP > 0 rows have a meaningful savings value
T = T(T.DSPCount > 0, :);
T.savings     = T.LC_dsp0 - T.LC;
T.savings_per_dsp = T.savings ./ T.DSPCount;

% Group by (IB, DSPCount) only — WB does not affect LC in DSP mode
% (weights are in ROM; SB_MAC16 width is fixed hardware)
T.savings_per_dsp = (T.LC_dsp0 - T.LC) ./ T.DSPCount;

ib_vals  = unique(T.InBits)';
dsp_vals = unique(T.DSPCount(T.DSPCount > 0))';

dsp_rows = {};
for i = ib_vals
  for d = dsp_vals
    mask = T.InBits==i & T.DSPCount==d;
    if sum(mask) < 4, continue; end
    oc_s = T.OutCh(mask); ic_s = T.InCh(mask); lc_s = T.LC(mask);
    X = [oc_s.*ic_s, oc_s, ic_s, ones(sum(mask),1)];

    if rank(X) < size(X, 2)
        % Rank-deficient: drop OC·IC interaction term
        X = [oc_s, ic_s, ones(sum(mask),1)];
        c_full = [0; X \ lc_s];
        flag = '*';
    else
        c_full = X \ lc_s;
        flag = '';
    end

    r2 = 1 - sum((lc_s - X*c_full(end-size(X,2)+1:end)).^2) / ...
             sum((lc_s - mean(lc_s)).^2);
    fprintf('IB=%d DSP=%d%s  LC=%.2f·OC·IC + %.2f·OC + %.2f·IC + %.2f  R²=%.3f  (N=%d)\n', ...
        i, d, flag, c_full(1), c_full(2), c_full(3), c_full(4), r2, sum(mask));
    % Store WB=-1 as sentinel: profile.py must look up DSP>0 corners by (IB, -1, DSP)
    dsp_rows{end+1} = {i, -1, d, 'LC', c_full(1), c_full(2), c_full(3), c_full(4), r2};
  end
end

%% Append DSP>0 corners to profile_coeffs.csv
dsp_T = cell2table(vertcat(dsp_rows{:}), 'VariableNames', ...
    {'InBits','WeightBits','DSPCount','Model','A','B','C','D','R2'});

coeffs_path = '../profiles/profile_coeffs.csv';
if isfile(coeffs_path)
    existing = readtable(coeffs_path);
    combined = [existing; dsp_T];
else
    combined = dsp_T;
end
writetable(combined, coeffs_path);
fprintf('Appended %d DSP>0 corners; total %d rows in profile_coeffs.csv\n', ...
    height(dsp_T), height(combined));

%% Plot 1: LC vs OutChannels/DSPCount (aggregated trend)
figure(1); clf;
T.oc_per_dsp = T.OutCh ./ T.DSPCount;

scatter(T.oc_per_dsp, T.LC, 20, T.InCh, 'filled');
xlabel('OutChannels / DSPCount');
ylabel('LC');
title('LC vs output channels per DSP block');
colorbar; cb = colorbar; cb.Label.String = 'InCh';

%% Plot 2: FF vs OutChannels/DSPCount (aggregated trend, same scale as LC)
figure(2); clf;

scatter(T.oc_per_dsp, T.FF, 20, T.InCh, 'filled');
xlabel('OutChannels / DSPCount');
ylabel('FF');
title('FF vs output channels per DSP block');
colorbar; cb = colorbar; cb.Label.String = 'InCh';
ylim([0, 5280]);   % match LC axis — shows FF is well under the bottleneck

%% Plot 3: LC vs DSPCount, one line per OutCh
% Average across (IC, IB, WB) within each (OC, DSPCount) slice
figure(3); clf; hold on;
T2 = T(T.InBits==4 & T.WeightBits==4, :);
T2.lc_frac = T2.LC ./ T2.LC_dsp0;   % 1.0 = no savings, lower = more savings

colors = lines(DSP_CAP);
for d = 1:DSP_CAP
    mask = T2.DSPCount == d;
    if sum(mask) < 2, continue; end
    [ratio, idx] = sort(T2.OutCh(mask) ./ d);
    frac = T2.lc_frac(mask); frac = frac(idx);
    plot(ratio, frac, '-o', 'Color', colors(d,:), 'DisplayName', sprintf('DSP=%d', d));
end

xlabel('OutChannels / DSPCount');
ylabel('LC / LC_{DSP=0}  (1.0 = no savings)');
title('Fractional LC vs OC/DSP ratio  (IB=4 WB=4, all IC)');
yline(1.0, 'k--');
legend('Location','best');



%% Summary
fprintf('Global mean savings/DSP : %.2f LC\n', mean(T.savings_per_dsp));
fprintf('Std deviation           : %.2f LC  (%.1f%%)\n', ...
    std(T.savings_per_dsp), 100*std(T.savings_per_dsp)/mean(T.savings_per_dsp));
fprintf('Min / Max               : %.1f / %.1f LC\n', ...
    min(T.savings_per_dsp), max(T.savings_per_dsp));
