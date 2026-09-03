function setup_cvx_mosek()
%SETUP_CVX_MOSEK  Add CVX and MOSEK to the MATLAB path for this session.
% IMPLEMENTATION CHOICE: paths match this machine's installs.

cvx_root = 'C:\Users\Nandu\Downloads\ABDM\Compressed\cvx\cvx';
mosek_tbx = 'C:\Program Files\Mosek\11.2\toolbox\R2019b';

if exist(fullfile(cvx_root, 'cvx_startup.m'), 'file') ~= 2
    error('CVX not found at %s', cvx_root);
end
if exist(fullfile(mosek_tbx, 'mosekopt.m'), 'file') ~= 2
    error('MOSEK toolbox not found at %s', mosek_tbx);
end

addpath(mosek_tbx);
run(fullfile(cvx_root, 'cvx_startup.m'));
cvx_solver mosek;
end
