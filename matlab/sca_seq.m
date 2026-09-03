function out = sca_seq(in)
%SCA_SEQ  Algorithm 1 SCA of Problem (P) in MATLAB CVX + MOSEK.
%
% Paper: each iteration solves one convexified (P) for UAV positions AND
% bandwidth (first-order Taylor of (3)–(5) and (25)). a_ij, b_ij are not
% in Alg. 1's update list and stay at initialization.
% PAPER CORRECTION: the PDF while-condition is inverted; stop when
% |ObjP(n)-ObjP(n-1)| <= epsilon after an update.
% stop_reason uses classify_stop_reason (same rule as Python):
%   accepted_steps==0 → STEP_SIZE_LIMIT (never CONVERGED);
%   CONVERGED requires accepted_steps>=1;
%   MAX_ITERATIONS only if the cap is hit with step still above min_step.
% After the joint solve, bandwidth is refreshed with the exact frozen-q LP
% so the next linearization point is true-feasible. Python evaluate() is
% the published score.

iot = reshape_xyz(in.iot_xyz);
uav = reshape_xyz(in.uav_xyz);
in.members = as_member_cell(in.members);
A = as_matrix(in.A, double(in.I), double(in.J));
proc = as_matrix(in.processing, double(in.I), double(in.J));
p = in.params;
step = double(in.step_m);
min_step = double(in.min_step_m);
shrink = double(in.step_shrink);
maxit = double(in.max_iterations);
eps_obj = double(in.epsilon);
improve_tol = double(in.improvement_tolerance);
fd_step = 1e-3;
if isfield(in, 'fd_step_m')
    fd_step = double(in.fd_step_m);
end

I = size(A, 1);
J = size(A, 2);
j_assoc = zeros(I, 1);
for i = 1:I
    [~, j_assoc(i)] = max(A(i, :));
end

SE = channel_se(iot, uav, p);
se_diff = NaN;
if isfield(in, 'se_python')
    se_py = as_matrix(in.se_python, I, J);
    se_diff = max(abs(SE(:) - se_py(:)));
end

floors = bandwidth_floors(A, SE, j_assoc, in.slacks(:), p);
solB = bandwidth_lp(pack_bw(A, SE, floors, p));
if solB.infeasible
    out = fail_out('init_bandwidth_infeasible', solB.status);
    out.se_max_abs_diff = se_diff;
    return
end
B = reshape(double(solB.B(:)), [I, J]);
B = sanitize_B(B, A, p);
[true0, feas0, info0] = true_metrics(iot, uav, A, proc, B, j_assoc, p, in);
if ~feas0
    out = fail_out('init_true_infeasible', solB.status);
    out.se_max_abs_diff = se_diff;
    return
end

hist = {new_hist_row(0, true0, true0, 1, step, 'init', solB.status, info0, info0)};
best_uav = uav;
best_B = B;
best_true = true0;
best_info = info0;
n_acc = 0;
n_rej = 0;
n_red = 0;
n_lp = 1;
stop_reason = 'MAX_ITERATIONS';
last_status = solB.status;
n_done = 0;

for n = 1:maxit
    n_done = n;
    if step <= min_step
        if n_acc == 0
            stop_reason = 'STEP_SIZE_LIMIT';
        else
            stop_reason = 'CONVERGED';
        end
        n_done = n - 1;
        break
    end
    SE = channel_se(iot, uav, p);
    [gx, gy] = se_jacobian_fd(iot, uav, p, fd_step);
    rfloors = rate_floors_bit(A, j_assoc, in.slacks(:), p);
    solP = convex_p(pack_joint(A, SE, gx, gy, B, rfloors, uav, step, p));
    n_lp = n_lp + 1;
    last_status = solP.status;
    if solP.infeasible
        n_rej = n_rej + 1;
        n_red = n_red + 1;
        used = step;
        step = step * shrink;
        hist{end + 1} = new_hist_row(n, best_true, NaN, 0, used, ...
            'convex_p_infeasible', solP.status, best_info, best_info); %#ok<AGROW>
        continue
    end
    xy = [double(solP.x(:)), double(solP.y(:))];
    move = max(abs(xy(:) - reshape(uav(:, 1:2), [], 1)));
    if move < min_step
        stop_reason = 'CONVERGED';
        hist{end + 1} = new_hist_row(n, best_true, best_true, 0, step, ...
            'stationary_convex_p', solP.status, best_info, best_info); %#ok<AGROW>
        break
    end
    % Alg. 1: update UAV positions, then bandwidth at the new geometry.
    uav_c = [xy, p.height * ones(J, 1)];
    SE_c = channel_se(iot, uav_c, p);
    floors_c = bandwidth_floors(A, SE_c, j_assoc, in.slacks(:), p);
    solB = bandwidth_lp(pack_bw(A, SE_c, floors_c, p));
    n_lp = n_lp + 1;
    last_status = solB.status;
    if solB.infeasible
        n_rej = n_rej + 1;
        n_red = n_red + 1;
        used = step;
        step = step * shrink;
        hist{end + 1} = new_hist_row(n, best_true, NaN, 0, used, ...
            'bandwidth_infeasible', solB.status, best_info, best_info); %#ok<AGROW>
        continue
    end
    B_c = sanitize_B(reshape(double(solB.B(:)), [I, J]), A, p);
    [true_c, feas_c, info_c] = true_metrics(iot, uav_c, A, proc, B_c, j_assoc, p, in);
    if ~feas_c || true_c <= best_true + improve_tol
        n_rej = n_rej + 1;
        n_red = n_red + 1;
        reason = 'true_infeasible';
        if feas_c
            reason = 'no_true_improvement';
        end
        used = step;
        step = step * shrink;
        hist{end + 1} = new_hist_row(n, best_true, true_c, 0, used, ...
            reason, solB.status, best_info, info_c); %#ok<AGROW>
        if n_acc > 0 && step <= min_step
            stop_reason = 'CONVERGED';
            break
        end
        continue
    end
    prev_true = best_true;
    prev_info = best_info;
    delta = true_c - best_true;
    uav = uav_c;
    B = B_c;
    best_uav = uav;
    best_B = B;
    best_true = true_c;
    best_info = info_c;
    n_acc = n_acc + 1;
    hist{end + 1} = new_hist_row(n, prev_true, true_c, 1, step, ...
        'accepted', solB.status, prev_info, info_c); %#ok<AGROW>
    if abs(delta) <= eps_obj
        stop_reason = 'CONVERGED';
        break
    end
end

% Must match Python uavdt.sca.algorithm.classify_stop_reason.
stop_reason = classify_stop_reason(stop_reason, n_acc, step, min_step);

[~, feas_f, info_f] = true_metrics(iot, best_uav, A, proc, best_B, j_assoc, p, in);
out = struct();
out.stop_reason = stop_reason;
out.cvx_status = last_status;
out.true_obj = best_true;
out.init_obj = true0;
out.accepted_steps = n_acc;
out.rejected_steps = n_rej;
out.step_size_reductions = n_red;
out.n_iterations = n_done;
out.n_lp_solves = n_lp;
out.final_step_m = step;
out.uav_xyz = best_uav(:);
out.B = best_B(:);
out.I = I;
out.J = J;
out.feasible = double(feas_f);
out.max_AoDT = info_f.max_AoDT;
out.min_sep = info_f.min_sep;
out.rho = info_f.rho(:);
out.history = hist;
out.infeasible = 0;
out.se_max_abs_diff = se_diff;
out.solver_name = 'MATLAB-CVX-MOSEK';
end


function floors = bandwidth_floors(A, SE, j_assoc, slacks, p)
I = size(A, 1);
J = size(A, 2);
floors = zeros(I, J);
for i = 1:I
    j = j_assoc(i);
    se = SE(i, j);
    slack = slacks(i);
    if se <= 1e-15 || ~(isfinite(slack)) || slack <= 1e-12
        floors(i, j) = inf;
        continue
    end
    floors(i, j) = max(p.R_min / se, p.S / (se * slack));
end
end


function floors = rate_floors_bit(A, j_assoc, slacks, p)
% Linearized (25) and (31): rhat >= max(R_min, S/slack) on associated links.
I = size(A, 1);
J = size(A, 2);
floors = -1e12 * ones(I, J);
for i = 1:I
    j = j_assoc(i);
    slack = slacks(i);
    if ~(isfinite(slack)) || slack <= 1e-12
        floors(i, j) = inf;
        continue
    end
    floors(i, j) = max(p.R_min, p.S / slack);
end
end


function [rate, feas, info] = true_metrics(iot, uav, A, proc, B, j_assoc, p, in)
SE = channel_se(iot, uav, p);
I = size(A, 1);
J = size(A, 2);
r = zeros(I, 1);
for i = 1:I
    r(i) = B(i, j_assoc(i)) * SE(i, j_assoc(i));
end
rate = sum(r);
qos = sum(r < p.R_min - 1e-6);
D = zeros(I, 1);
for i = 1:I
    D(i) = p.S / max(r(i), 1e-30);
    if proc(i, j_assoc(i)) < 0.5
        D(i) = D(i) + p.t_u2u;
    end
end
members = in.members;
Q = double(in.Q(:));
K = numel(Q);
aodt = zeros(K, 1);
for k = 1:K
    idx = double(members{k}(:)) + 1;
    aodt(k) = max(D(idx)) + Q(k);
end
aodt_v = sum(aodt > p.T_k + 1e-9);
sep_v = 0;
min_sep = inf;
for j = 1:J
    for k = (j + 1):J
        d = norm(uav(j, :) - uav(k, :));
        min_sep = min(min_sep, d);
        if d < p.theta - 1e-9
            sep_v = sep_v + 1;
        end
    end
end
bw_ok = sum(B(:)) <= p.B_sys + 1e-3;
in_field = all(uav(:, 1) >= -1e-9) && all(uav(:, 1) <= p.area_x + 1e-9) ...
    && all(uav(:, 2) >= -1e-9) && all(uav(:, 2) <= p.area_y + 1e-9);
rho = zeros(J, 1);
lam = double(in.lambdas(:));
for j = 1:J
    rho(j) = sum(proc(:, j) .* lam) / p.mu;
end
cpu_v = sum(rho >= 1 - 1e-12);
if ~isfinite(min_sep)
    min_sep = 0;
end
feas = (qos == 0) && (aodt_v == 0) && (sep_v == 0) && bw_ok && in_field && (cpu_v == 0);
info = struct('max_AoDT', max(aodt), 'min_sep', min_sep, 'rho', rho, ...
    'qos', qos, 'aodt_v', aodt_v, 'bw_sum', sum(B(:)), 'max_rho', max(rho));
end


function s = pack_bw(A, SE, floors, p)
s = struct();
s.I = size(A, 1);
s.J = size(A, 2);
s.A = A(:);
s.SE = SE(:);
s.floors = floors(:);
s.B_sys = p.B_sys;
if isfield(p, 'B_cap')
    s.B_cap = p.B_cap;
else
    s.B_cap = p.B_sys;
end
end


function s = pack_joint(A, SE, gx, gy, B0, rate_floors, uav, step, p)
s = struct();
s.I = size(A, 1);
s.J = size(A, 2);
s.A = A(:);
s.SE = SE(:);
s.gx = gx(:);
s.gy = gy(:);
s.B0 = B0(:);
s.rate_floors = rate_floors(:);
s.x0 = uav(:, 1);
s.y0 = uav(:, 2);
s.step_m = step;
s.area_x = p.area_x;
s.area_y = p.area_y;
s.theta = p.theta;
s.B_sys = p.B_sys;
if isfield(p, 'B_cap')
    s.B_cap = p.B_cap;
else
    s.B_cap = p.B_sys;
end
end


function xyz = reshape_xyz(v)
v = double(v);
if size(v, 2) == 3 && size(v, 1) >= 1 && ~isscalar(v)
    xyz = v;
    return
end
v = v(:);
n = numel(v) / 3;
xyz = reshape(v, [n, 3]);
end


function M = as_matrix(v, I, J)
v = double(v);
if isequal(size(v), [I, J])
    M = v;
    return
end
M = reshape(v(:), [I, J]);
end


function members = as_member_cell(raw)
if iscell(raw)
    members = raw;
    return
end
tmpm = double(raw);
members = cell(size(tmpm, 1), 1);
for k = 1:size(tmpm, 1)
    members{k} = tmpm(k, :);
end
end


function B = sanitize_B(B, A, p)
B = max(B, 0);
B = B .* (A > 0.5);
if isfield(p, 'B_cap')
    B = min(B, p.B_cap);
end
s = sum(B(:));
if s > p.B_sys + 1e-6
    B = B * (p.B_sys / s);
end
end


function row = new_hist_row(it, current, cand, acc, step, reason, st, info_cur, info_cand)
row = struct();
row.iteration = it;
row.current_true_objective = current;
if isfinite(cand)
    row.candidate_true_objective = cand;
else
    row.candidate_true_objective = [];
end
row.current_max_AoDT = info_cur.max_AoDT;
row.candidate_max_AoDT = info_cand.max_AoDT;
row.current_min_separation = info_cur.min_sep;
row.candidate_min_separation = info_cand.min_sep;
row.step_size = step;
row.bandwidth_objective = [];
if acc && isfinite(cand)
    row.true_objective = cand;
else
    row.true_objective = current;
end
row.accepted = acc;
row.rejection_reason = reason;
row.solver_status = st;
row.qos_violations = info_cand.qos;
row.aodt_violations = info_cand.aodt_v;
row.max_cpu_load = info_cand.max_rho;
row.bandwidth_usage = info_cur.bw_sum;
row.solve_time_s = 0;
end


function reason = classify_stop_reason(proposed, n_acc, step, min_step)
%CLASSIFY_STOP_REASON  Same rule as Python classify_stop_reason.
%
% accepted_steps == 0 at any stop → STEP_SIZE_LIMIT (never CONVERGED),
% except init_* and MAX_ITERATIONS (cap hit with step still above floor).
% CONVERGED requires accepted_steps >= 1.
if numel(proposed) >= 5 && strcmp(proposed(1:5), 'init_')
    reason = proposed;
    return
end
if strcmp(proposed, 'MAX_ITERATIONS') && n_acc > 0 && step <= min_step
    reason = 'CONVERGED';
    return
end
if n_acc == 0 && ~strcmp(proposed, 'MAX_ITERATIONS')
    if step <= min_step || strcmp(proposed, 'CONVERGED')
        reason = 'STEP_SIZE_LIMIT';
        return
    end
end
reason = proposed;
end


function out = fail_out(reason, status)
out = struct();
out.stop_reason = reason;
out.cvx_status = status;
out.infeasible = 1;
out.true_obj = 0;
out.init_obj = 0;
out.accepted_steps = 0;
out.rejected_steps = 0;
out.step_size_reductions = 0;
out.n_iterations = 0;
out.n_lp_solves = 0;
out.final_step_m = 0;
out.feasible = 0;
out.history = {};
out.solver_name = 'MATLAB-CVX-MOSEK';
out.se_max_abs_diff = NaN;
end
