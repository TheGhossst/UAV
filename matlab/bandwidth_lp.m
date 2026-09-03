function out = bandwidth_lp(in)
%BANDWIDTH_LP  Exact bandwidth LP at frozen UAV geometry.
%
% Sequential SCA-style convex subproblem (IMPLEMENTATION CHOICE).
% Physics (SE, association, floors) come from the Python core.

I = double(in.I);
J = double(in.J);
A = reshape(double(in.A(:)), [I, J]);
SE = reshape(double(in.SE(:)), [I, J]);
floors = reshape(double(in.floors(:)), [I, J]);
Bsys = double(in.B_sys);
if isfield(in, 'B_cap')
    Bcap = double(in.B_cap);
else
    Bcap = Bsys;
end
if any(~isfinite(floors(:)))
    out = struct();
    out.status = 'aodt_slack_nonpositive';
    out.objective = 0;
    out.I = I;
    out.J = J;
    out.B = zeros(I * J, 1);
    out.infeasible = 1;
    return
end

cvx_begin quiet
    cvx_solver mosek
    variable B(I, J)
    maximize(sum(sum(A .* SE .* B)))
    subject to
        B >= 0;
        B <= Bcap * A;
        sum(sum(B)) <= Bsys;
        B >= floors;
cvx_end

out = struct();
out.status = cvx_status;
out.objective = cvx_optval;
if ~(isfinite(out.objective))
    out.objective = 0;
end
out.I = I;
out.J = J;
if exist('B', 'var') && ~isempty(B)
    out.B = full(double(B(:)));
else
    out.B = zeros(I * J, 1);
end
if any(strcmpi(cvx_status, {'Solved', 'Inaccurate/Solved'}))
    out.infeasible = 0;
else
    out.infeasible = 1;
end
end
