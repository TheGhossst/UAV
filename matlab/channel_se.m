function SE = channel_se(iot_xyz, uav_xyz, p)
%CHANNEL_SE  Spectral efficiency SE_ij = log2(1+SNR) from paper Eqs. (1)–(6).
% Must match uavdt.channel (frozen Python core). Used only inside MATLAB SCA.

I = size(iot_xyz, 1);
J = size(uav_xyz, 1);
H = uav_xyz(1, 3);
SE = zeros(I, J);
jfs = 20 * log10(p.f_c) + 20 * log10(4 * pi / p.c_light);
noise = p.sigma ^ 2;
for i = 1:I
    for j = 1:J
        d = norm(iot_xyz(i, :) - uav_xyz(j, :));
        d = max(d, 1e-15);
        arg = min(max(H / d, 0), 1);
        theta = asin(arg);  % radians, as written in Eq. (4)
        expo = -p.env_b * (theta - p.env_a);
        expo = min(max(expo, -80), 80);
        pLoS = 1 / (1 + p.env_a * exp(expo));
        Llos = jfs + 20 * log10(d) + p.eta_los;
        Lnlos = jfs + 20 * log10(d) + p.eta_nlos;
        Lavg = pLoS * Llos + (1 - pLoS) * Lnlos;
        prx = p.p_i * 10 ^ (-Lavg / 10);
        snr = prx / noise;
        SE(i, j) = log2(1 + max(snr, 0));
    end
end
end
