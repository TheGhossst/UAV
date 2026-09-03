function [x, ok, cvx_status_out] = convexified_lp(c, A_ub, b_ub, lb, ub)
% CONVEXIFIED_LP  min c'*x  s.t.  A_ub*x <= b_ub,  lb <= x <= ub
%
% Same linear program as src/solvers/sca.py:_assemble_convexified_lp.
% Solved with CVX + MOSEK (IEEE TNSM 2026 §V stack).

    c = c(:);
    b_ub = b_ub(:);
    lb = lb(:);
    ub = ub(:);
    n = numel(c);
    if isempty(A_ub)
        A_ub = zeros(0, n);
    end

    x = nan(n, 1);
    ok = 0;
    cvx_status_out = 'error';

    if exist('cvx_begin', 'file') ~= 2
        error(['CVX is not on the MATLAB path. In MATLAB run cvx_setup, ' ...
               'then retry.']);
    end

    finite_lb = isfinite(lb);
    finite_ub = isfinite(ub);

    cvx_begin quiet
        cvx_solver mosek
        variable x_var(n)
        minimize( c' * x_var )
        subject to
            if ~isempty(A_ub)
                A_ub * x_var <= b_ub;
            end
            if any(finite_lb)
                x_var(finite_lb) >= lb(finite_lb);
            end
            if any(finite_ub)
                x_var(finite_ub) <= ub(finite_ub);
            end
    cvx_end

    cvx_status_out = cvx_status;
    solved = strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved');
    if solved
        x = x_var;
        ok = 1;
    end
end
