function [gx, gy] = se_jacobian_fd(iot_xyz, uav_xyz, p, step)
%SE_JACOBIAN_FD  Numerical dSE/dx, dSE/dy matching uavdt.sca.linearize.

if nargin < 4
    step = 1e-3;
end
SE0 = channel_se(iot_xyz, uav_xyz, p);
[I, J] = size(SE0);
gx = zeros(I, J);
gy = zeros(I, J);
lim = [p.area_x, p.area_y];
for u = 1:J
    for ax = 1:2
        x0 = uav_xyz(u, ax);
        plus = uav_xyz;
        minus = uav_xyz;
        plus(u, ax) = min(x0 + step, lim(ax));
        minus(u, ax) = max(x0 - step, 0);
        dp = plus(u, ax) - x0;
        dm = x0 - minus(u, ax);
        if dp > 1e-16 && dm > 1e-16
            SEp = channel_se(iot_xyz, plus, p);
            SEm = channel_se(iot_xyz, minus, p);
            g = (SEp(:, u) - SEm(:, u)) / (plus(u, ax) - minus(u, ax));
        elseif dp > 1e-16
            SEp = channel_se(iot_xyz, plus, p);
            g = (SEp(:, u) - SE0(:, u)) / dp;
        else
            SEm = channel_se(iot_xyz, minus, p);
            g = (SE0(:, u) - SEm(:, u)) / dm;
        end
        if ax == 1
            gx(:, u) = g;
        else
            gy(:, u) = g;
        end
    end
end
end
