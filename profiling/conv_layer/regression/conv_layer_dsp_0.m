%% conv_layer_dsp_0.m — Fit LC and FF models from conv_layer_dsp_0.csv per (InBits, WeightBits) slice
%% and extrapolate to predict the combinational (DSP=0) case, which is too large to synthesize in many corners of the parameter space.
% iCE40 UP5K caps: 5280 LC, 3520 FF
% LC model per slice: A*OC*IC + B*OC + C*IC + D
% FF model per slice: A*OC*log2(IC) + B*OC + C*IC + D

LC_CAP = 5280;
FF_CAP = 3520;

%% Load
T = readtable('../profiles/sweep_conv_dsp_0.csv');
oc = T.OutCh;
ic = T.InCh;
ib = T.InBits;
wb = T.WeightBits;
lc = T.LC;
ff = T.FF;

ib_vals = unique(ib)';
wb_vals = unique(wb)';

%% Per-(IB,WB) slice fit
lc_fits = [];   % [IB, WB, A, B, C, D, R2, N]
ff_fits = [];

for i = ib_vals
  for w = wb_vals
    mask = ib == i & wb == w;
    n    = sum(mask);
    if n < 4, continue; end

    oc_s = oc(mask);  ic_s = ic(mask);
    lc_s = lc(mask);  ff_s = ff(mask);

    % LC fit
    X_lc = [oc_s.*ic_s, oc_s, ic_s, ones(n,1)];
    c_lc = X_lc \ lc_s;
    r2_lc = r2score(lc_s, X_lc * c_lc);
    lc_fits(end+1,:) = [i, w, c_lc', r2_lc, n];

    % FF fit
    X_ff = [oc_s.*log2(max(ic_s,1)), oc_s, ic_s, ones(n,1)];
    c_ff = X_ff \ ff_s;
    r2_ff = r2score(ff_s, X_ff * c_ff);
    ff_fits(end+1,:) = [i, w, c_ff', r2_ff, n];
  end
end

LC_T = array2table(lc_fits, 'VariableNames', ...
    {'InBits','WeightBits','A','B','C','D','R2','N'});
FF_T = array2table(ff_fits, 'VariableNames', ...
    {'InBits','WeightBits','A','B','C','D','R2','N'});

disp('=== LC fit per (InBits, WeightBits) ===');  disp(LC_T)
disp('=== FF fit per (InBits, WeightBits) ===');  disp(FF_T)

%% Heatmap: effective MAC cost A vs (IB, WB)
n_ib = numel(ib_vals);
n_wb = numel(wb_vals);
A_lc = NaN(n_ib, n_wb);
R2_lc = NaN(n_ib, n_wb);

for k = 1:height(LC_T)
    ri = find(ib_vals == LC_T.InBits(k));
    ci = find(wb_vals == LC_T.WeightBits(k));
    A_lc(ri,ci)  = LC_T.A(k);
    R2_lc(ri,ci) = LC_T.R2(k);
end

figure(1); clf;
heatmap(wb_vals, ib_vals, A_lc, ...
    'Title','LC: effective MAC cost A (LCs per OC·IC)', ...
    'XLabel','WeightBits', 'YLabel','InBits', ...
    'Colormap', parula, 'MissingDataColor',[0.85 0.85 0.85]);

figure(2); clf;
heatmap(wb_vals, ib_vals, R2_lc, ...
    'Title','LC fit R² per (InBits, WeightBits)', ...
    'XLabel','WeightBits', 'YLabel','InBits', ...
    'Colormap', parula, 'ColorLimits',[0 1], ...
    'MissingDataColor',[0.85 0.85 0.85]);

%% Heatmap: maximum feasible OC*IC product at each (IB, WB)
max_oc_ic = NaN(n_ib, n_wb);
for k = 1:height(LC_T)
    % Solve A*x + D <= LC_CAP for OC=IC=sqrt(x) (symmetric worst case)
    % More useful: max OC*IC s.t. A*OC*IC + B*OC + C*IC + D <= LC_CAP
    % Approximate by setting OC=IC=sqrt(x) and ignoring B,C terms
    A = LC_T.A(k);  D = LC_T.D(k);
    if A > 0
        max_prod = (LC_CAP - D) / A;
        ri = find(ib_vals == LC_T.InBits(k));
        ci = find(wb_vals == LC_T.WeightBits(k));
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
title(t, 'LC: predicted vs actual per slice');

for p = 1:size(check_pairs,1)
    i = check_pairs(p,1);  w = check_pairs(p,2);
    row = LC_T(LC_T.InBits==i & LC_T.WeightBits==w, :);
    if isempty(row), nexttile; continue; end

    mask  = ib==i & wb==w;
    oc_s  = oc(mask); ic_s = ic(mask); lc_s = lc(mask);
    X     = [oc_s.*ic_s, oc_s, ic_s, ones(sum(mask),1)];
    pred  = X * [row.A; row.B; row.C; row.D];

    nexttile;
    scatter(lc_s, pred, 18, 'filled'); hold on;
    lim = [0, LC_CAP * 1.05];
    plot(lim, lim, 'r--');
    xlim(lim); ylim(lim);
    xlabel('Actual LC'); ylabel('Predicted LC');
    title(sprintf('IB=%d WB=%d  R²=%.3f', i, w, row.R2));
end

%% Export coefficients to profile_coeffs.csv
rows = {};
for k = 1:height(LC_T)
    rows{end+1} = {LC_T.InBits(k), LC_T.WeightBits(k), 0, 'LC', ...
                   LC_T.A(k), LC_T.B(k), LC_T.C(k), LC_T.D(k), LC_T.R2(k)};
end
for k = 1:height(FF_T)
    rows{end+1} = {FF_T.InBits(k), FF_T.WeightBits(k), 0, 'FF', ...
                   FF_T.A(k), FF_T.B(k), FF_T.C(k), FF_T.D(k), FF_T.R2(k)};
end
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
