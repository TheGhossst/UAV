function out = convex_p(in)
%CONVEX_P  Joint convexified Problem (P) for Algorithm 1.
%
% Variables: UAV (x,y) and B_ij. a_ij, b_ij held fixed (Alg. 1 updates
% only positions and bandwidth).
%
% Paper §V: first-order Taylor of (3)–(5) through SE_ij(q), and of the
% product r_ij = B_ij SE_ij(q) used in (25).
% Affine surrogate:
%   rhat = SE0 .* B + B0 .* (gx .* dx + gy .* dy)
% (25) and (31) at this map: rhat >= max(R_min, S/slack) on associated links.
% (28) supporting halfspace. Field box. L-inf neighborhood of q0.

I = double(in.I);
J = double(in.J);
A = reshape(double(in.A(:)), [I, J]);
SE = reshape(double(in.SE(:)), [I, J]);
gx = reshape(double(in.gx(:)), [I, J]);
gy = reshape(double(in.gy(:)), [I, J]);
B0 = reshape(double(in.B0(:)), [I, J]);
rate_floors = reshape(double(in.rate_floors(:)), [I, J]);
x0 = double(in.x0(:));
y0 = double(in.y0(:));
step = double(in.step_m);
area_x = double(in.area_x);
area_y = double(in.area_y);
theta = double(in.theta);
Bsys = double(in.B_sys);
if isfield(in, 'B_cap')
    Bcap = double(in.B_cap);
else
    Bcap = Bsys;
end

if any(~isfinite(rate_floors(A > 0.5)))
    out = struct();
    out.status = 'aodt_slack_nonpositive';
    out.objective = 0;
    out.x = x0;
    out.y = y0;
    out.B = zeros(I * J, 1);
    out.infeasible = 1;
    return
end

cvx_begin quiet
    cvx_solver mosek
    variable x(J)
    variable y(J)
    variable B(I, J)
    dx = x - x0;
    dy = y - y0;
    rhat = SE .* B + B0 .* (gx .* (ones(I, 1) * dx.') + gy .* (ones(I, 1) * dy.'));
    maximize(sum(sum(A .* rhat)) - 1e-6 * sum(abs(x - x0) + abs(y - y0)))
    subject to
        B >= 0;
        B <= Bcap * A;
        sum(sum(B)) <= Bsys;
        x >= 0;
        x <= area_x;
        y >= 0;
        y <= area_y;
        abs(x - x0) <= step;
        abs(y - y0) <= step;
        rhat >= rate_floors;
        for p = 1:J
            for q = (p + 1):J
                z0x = x0(p) - x0(q);
                z0y = y0(p) - y0(q);
                nrm = hypot(z0x, z0y);
                if nrm < 1e-12
                    error('UAVs %d and %d coincide; cannot linearize (28)', p, q);
                end
                ux = z0x / nrm;
                uy = z0y / nrm;
                ux * (x(p) - x(q)) + uy * (y(p) - y(q)) >= theta;
            end
        end
cvx_end

out = struct();
out.status = cvx_status;
out.objective = cvx_optval;
if ~(isfinite(out.objective))
    out.objective = 0;
end
if exist('x', 'var') && ~isempty(x)
    out.x = full(double(x(:)));
    out.y = full(double(y(:)));
else
    out.x = x0;
    out.y = y0;
end
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
