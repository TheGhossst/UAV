function out = position_step_lp(in)
%POSITION_STEP_LP  Convex position step with bandwidth held fixed.
%
% B frozen. Objective is first-order in q of sum a B SE(q).
% Not used as a QoS certificate.

I = double(in.I);
J = double(in.J);
A = reshape(double(in.A(:)), [I, J]);
SE = reshape(double(in.SE(:)), [I, J]);
gx = reshape(double(in.gx(:)), [I, J]);
gy = reshape(double(in.gy(:)), [I, J]);
B0 = reshape(double(in.B0(:)), [I, J]);
x0 = double(in.x0(:));
y0 = double(in.y0(:));
step = double(in.step_m);
area_x = double(in.area_x);
area_y = double(in.area_y);
theta = double(in.theta);

cvx_begin quiet
    cvx_solver mosek
    variable x(J)
    variable y(J)
    dx = x - x0;
    dy = y - y0;
    SEhat = SE + gx .* (ones(I, 1) * dx.') + gy .* (ones(I, 1) * dy.');
    maximize(sum(sum(A .* B0 .* SEhat)))
    subject to
        x >= 0;
        x <= area_x;
        y >= 0;
        y <= area_y;
        abs(x - x0) <= step;
        abs(y - y0) <= step;
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
out.x = full(x(:));
out.y = full(y(:));
if any(strcmpi(cvx_status, {'Solved', 'Inaccurate/Solved'}))
    out.infeasible = 0;
else
    out.infeasible = 1;
end
end
