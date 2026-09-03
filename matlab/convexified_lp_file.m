function ok = convexified_lp_file(in_mat, out_mat)
% CONVEXIFIED_LP_FILE  Load an LP from IN_MAT, solve with CVX+MOSEK, save OUT_MAT.
%
% IN_MAT fields: c, A_ub, b_ub, lb, ub  (column vectors / matrix)
% OUT_MAT fields: x, ok, st

    S = load(in_mat);
    [x, ok, st] = convexified_lp(S.c, S.A_ub, S.b_ub, S.lb, S.ub);
    save(out_mat, 'x', 'ok', 'st');
end
