%% conv_layer_dsp_0.m — Fit LUT4 and FF models from conv_layer_dsp_0.csv per (InBits, WeightBits) slice
%% and extrapolate to predict the combinational (DSP=0) case, which is too large to synthesize in many corners of the parameter space.
% iCE40 Icebreaker V1.1a caps: 5280 LUT4, 3520 FF
% LUT4 model per slice: A*OC*IC + B*OC + C*IC + D
% FF model per slice: A*OC*log2(IC) + B*OC + C*IC + D

LC_CAP = 5280;

%% Load
T = readtable('../profiles/sweep_conv_dsp_0.csv');
oc = T.OutCh;
ic = T.InCh;
ib = T.InBits;
wb = T.WeightBits;
lut4 = T.LUT4;
ff = T.FF;

ib_vals = unique(ib)';
wb_vals = unique(wb)';

%% Per-(IB,WB) slice fit
max_slices = numel(ib_vals) * numel(wb_vals);
lut4_fits = NaN(max_slices, 8);   % [IB, WB, A, B, C, D, R2, N]
ff_fits = NaN(max_slices, 8);
fit_idx = 0;

for i = ib_vals
  for w = wb_vals
    mask = ib == i & wb == w;
    n    = sum(mask);
    if n < 4, continue; end

    oc_s = oc(mask);  ic_s = ic(mask);
    lut4_s = lut4(mask);  ff_s = ff(mask);

    % LUT4 fit
    X_lc = [oc_s.*ic_s, oc_s, ic_s, ones(n,1)];
    c_lc = X_lc \ lut4_s;
    r2_lc = r2score(lut4_s, X_lc * c_lc);

    % FF fit
    X_ff = [oc_s.*log2(max(ic_s,1)), oc_s, ic_s, ones(n,1)];
    c_ff = X_ff \ ff_s;
    r2_ff = r2score(ff_s, X_ff * c_ff);

        fit_idx = fit_idx + 1;
        lut4_fits(fit_idx,:) = [i, w, c_lc', r2_lc, n];
        ff_fits(fit_idx,:) = [i, w, c_ff', r2_ff, n];
  end
end

lut4_fits = lut4_fits(1:fit_idx,:);
ff_fits = ff_fits(1:fit_idx,:);

LUT4_T = array2table(lut4_fits, 'VariableNames', ...
    {'InBits','WeightBits','A','B','C','D','R2','N'});
FF_T = array2table(ff_fits, 'VariableNames', ...
    {'InBits','WeightBits','A','B','C','D','R2','N'});

disp('=== LUT4 fit per (InBits, WeightBits) ===');  disp(LUT4_T)
disp('=== FF fit per (InBits, WeightBits) ===');  disp(FF_T)

%% Heatmap: effective MAC cost A vs (IB, WB)
n_ib = numel(ib_vals);
n_wb = numel(wb_vals);
A_lc = NaN(n_ib, n_wb);
R2_lc = NaN(n_ib, n_wb);

for k = 1:height(LUT4_T)
    ri = find(ib_vals == LUT4_T.InBits(k));
    ci = find(wb_vals == LUT4_T.WeightBits(k));
    A_lc(ri,ci)  = LUT4_T.A(k);
    R2_lc(ri,ci) = LUT4_T.R2(k);
end

figure(1); clf;
heatmap(wb_vals, ib_vals, A_lc, ...
    'Title','LUT4: effective MAC cost A (LCs per OC·IC)', ...
    'XLabel','WeightBits', 'YLabel','InBits', ...
    'Colormap', parula, 'MissingDataColor',[0.85 0.85 0.85]);

figure(2); clf;
heatmap(wb_vals, ib_vals, R2_lc, ...
    'Title','LUT4 fit R² per (InBits, WeightBits)', ...
    'XLabel','WeightBits', 'YLabel','InBits', ...
    'Colormap', parula, 'ColorLimits',[0 1], ...
    'MissingDataColor',[0.85 0.85 0.85]);

%% Heatmap: maximum feasible OC*IC product at each (IB, WB)
max_oc_ic = NaN(n_ib, n_wb);
for k = 1:height(LUT4_T)
    % Solve A*x + D <= LC_CAP for OC=IC=sqrt(x) (symmetric worst case)
    % More useful: max OC*IC s.t. A*OC*IC + B*OC + C*IC + D <= LC_CAP
    % Approximate by setting OC=IC=sqrt(x) and ignoring B,C terms
    A = LUT4_T.A(k);  D = LUT4_T.D(k);
    if A > 0
        max_prod = (LC_CAP - D) / A;
        ri = find(ib_vals == LUT4_T.InBits(k));
        ci = find(wb_vals == LUT4_T.WeightBits(k));
        max_oc_ic(ri,ci) = max(max_prod, 0);
    end
end

figure(3); clf;
heatmap(wb_vals, ib_vals, max_oc_ic, ...
    'Title','Max feasible OC×IC (approx, ignoring linear terms)', ...
    'XLabel','WeightBits', 'YLabel','InBits', ...
    'Colormap', parula, 'MissingDataColor',[0.85 0.85 0.85]);

%% Predicted vs actual for a few representative slices
check_pairs = [1 2; 2 4; 4 4; 8 8];
figure(4); clf; t = tiledlayout(2, 2, 'TileSpacing','compact');
title(t, 'LUT4: predicted vs actual per slice');

for p = 1:size(check_pairs,1)
    i = check_pairs(p,1);  w = check_pairs(p,2);
    row = LUT4_T(LUT4_T.InBits==i & LUT4_T.WeightBits==w, :);
    if isempty(row), nexttile; continue; end

    mask  = ib==i & wb==w;
    oc_s  = oc(mask); ic_s = ic(mask); lut4_s = lut4(mask);
    X     = [oc_s.*ic_s, oc_s, ic_s, ones(sum(mask),1)];
    pred  = X * [row.A; row.B; row.C; row.D];

    nexttile;
    scatter(lut4_s, pred, 18, 'filled'); hold on;
    lim = [0, LC_CAP * 1.05];
    plot(lim, lim, 'r--');
    xlim(lim); ylim(lim);
    xlabel('Actual LUT4'); ylabel('Predicted LUT4');
    title(sprintf('IB=%d WB=%d  R²=%.3f', i, w, row.R2));
end

%% Export coefficients to profile_coeffs.csv
rows = cell(height(LUT4_T) + height(FF_T), 1);
row_idx = 0;
for k = 1:height(LUT4_T)
    row_idx = row_idx + 1;
    rows{row_idx} = {LUT4_T.InBits(k), LUT4_T.WeightBits(k), 0, 'LUT4', ...
                     LUT4_T.A(k), LUT4_T.B(k), LUT4_T.C(k), LUT4_T.D(k), LUT4_T.R2(k)};
end
for k = 1:height(FF_T)
    row_idx = row_idx + 1;
    rows{row_idx} = {FF_T.InBits(k), FF_T.WeightBits(k), 0, 'FF', ...
                     FF_T.A(k), FF_T.B(k), FF_T.C(k), FF_T.D(k), FF_T.R2(k)};
end
rows = rows(1:row_idx);
out_T = cell2table(vertcat(rows{:}), 'VariableNames', ...
    {'InBits','WeightBits','DSPCount','Model','A','B','C','D','R2'});
writetable(out_T, '../profiles/profile_coeffs.csv');
fprintf('Exported %d DSP=0 corners to profile_coeffs.csv\n', height(out_T));

%% Helper
function s = r2score(y, yhat)
    ss_res = sum((y - yhat).^2);
    ss_tot = sum((y - mean(y)).^2);
    s = 1 - ss_res / ss_tot;
end
