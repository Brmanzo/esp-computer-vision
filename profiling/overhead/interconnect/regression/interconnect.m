%% interconnect.m — Fit LC overhead model from bitstream sweep results
%%
%% Full-design flat synthesis produces routing/glue LUT4s not captured by
%% per-module profiling.  Regression shows act_lc is driven almost entirely
%% by pred_ff (the pipeline register estimate), not pred_lut4:
%%
%%   act_lc = A·pred_ff + B     R² ≈ 0.97
%%
%% The pred_lut4 coefficient in a full model is ~0, confirming that the
%% arithmetic LUT estimates are irrelevant once DSPs absorb the MAC work.
%% pred_ff acts as a proxy for routing complexity (pipeline depth × width).
%%
%% Sources:  ../profiles/bitstream_results.txt
%% Exports:  ../profiles/interconnect_coeffs.csv

results_path = '../profiles/bitstream_results.txt';

%% ── Parse bitstream_results.txt ──────────────────────────────────────────────
fid = fopen(results_path, 'r');
assert(fid >= 0, 'Cannot open %s', results_path);

idx_v       = [];
pred_lut4_v = [];
pred_ff_v   = [];
act_lut4_v  = [];
act_ff_v    = [];
act_lc_v    = [];

while true
    line = fgetl(fid);
    if ~ischar(line), break; end

    %  idx  pred_lut4  act_lut4  lut4_err%  pred_ff  act_ff  ff_err%  act_lc  ok  arch
    tok = regexp(line, ...
        ['^\s*(\d+)\s+(\S+)\s+(\S+)\s+\S+\s+(\S+)\s+(\S+)\s+\S+\s+(\S+)' ...
         '\s+ok\s+(.*)'], 'tokens');
    if isempty(tok), continue; end
    t = tok{1};

    pred_lut4 = str2double(t{2});
    act_lut4  = str2double(t{3});
    pred_ff   = str2double(t{4});
    act_ff    = str2double(t{5});
    act_lc    = str2double(t{6});

    if any(isnan([pred_lut4, act_lut4, pred_ff, act_ff, act_lc])), continue; end

    idx_v       = [idx_v;       str2double(t{1})];
    pred_lut4_v = [pred_lut4_v; pred_lut4];
    pred_ff_v   = [pred_ff_v;   pred_ff];
    act_lut4_v  = [act_lut4_v;  act_lut4];
    act_ff_v    = [act_ff_v;    act_ff];
    act_lc_v    = [act_lc_v;    act_lc];
end
fclose(fid);

N = numel(act_lc_v);
fprintf('Loaded %d ok rows\n', N);
assert(N >= 3, 'Too few data points for regression (need >= 3)');

%% ── Model: act_lc = A·pred_ff + B ───────────────────────────────────────────
X  = [pred_ff_v, ones(N,1)];
c  = X \ act_lc_v;
r2 = r2score(act_lc_v, X * c);

A = c(1);
B = c(2);

fprintf('\nact_lc = %.4f·pred_ff + %.2f   R²=%.4f  (N=%d)\n', A, B, r2, N);

LC_CAP = 5280;
ff_cap = (LC_CAP - B) / A;
fprintf('Implied FF cap for feasibility: pred_ff <= %.1f\n', ff_cap);

%% ── Export ───────────────────────────────────────────────────────────────────
out_T = table({'A'; 'B'; 'R2'; 'N'}, [A; B; r2; N], 'VariableNames', {'Term', 'Value'});
writetable(out_T, '../profiles/interconnect_coeffs.csv');
fprintf('Exported to interconnect_coeffs.csv\n');

%% ── Plots ────────────────────────────────────────────────────────────────────
figure(1); clf;
pred_lc = X * c;
scatter(act_lc_v, pred_lc, 40, pred_ff_v, 'filled'); hold on;
lo = min([act_lc_v; pred_lc]) - 30;
hi = max([act_lc_v; pred_lc]) + 30;
plot([lo hi], [lo hi], 'r--');
xlim([lo hi]); ylim([lo hi]);
xlabel('Actual ICESTORM\_LC');
ylabel('Predicted LC  (A·pred\_ff + B)');
title(sprintf('act\\_lc = %.3f·pred\\_ff + %.1f   R²=%.4f  (N=%d)', A, B, r2, N));
cb = colorbar; cb.Label.String = 'pred\_ff';

figure(2); clf;
scatter(pred_ff_v, act_lc_v, 40, pred_lut4_v, 'filled'); hold on;
ff_range = linspace(min(pred_ff_v), max(pred_ff_v), 200)';
plot(ff_range, A*ff_range + B, 'r-', 'LineWidth', 1.5);
xlabel('pred\_ff');
ylabel('act\_lc');
title('act\_lc vs pred\_ff  (colour = pred\_lut4)');
cb = colorbar; cb.Label.String = 'pred\_lut4';

%% ── Helper ───────────────────────────────────────────────────────────────────
function s = r2score(y, yhat)
    ss_res = sum((y - yhat).^2);
    ss_tot = sum((y - mean(y)).^2);
    s = 1 - ss_res / ss_tot;
end
