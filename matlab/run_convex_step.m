function run_convex_step(in_path, out_path)
%RUN_CONVEX_STEP  Batch entry: read JSON, solve CVX/MOSEK, write JSON.
%
% Called from Python:
%   matlab -batch "addpath('matlab'); run_convex_step('in.json','out.json')"

setup_cvx_mosek();
raw = fileread(in_path);
in = jsondecode(raw);
task = char(in.task);

switch task
    case 'bandwidth'
        sol = bandwidth_lp(in);
    case 'position'
        sol = position_step_lp(in);
    case 'joint'
        sol = convex_p(in);
    otherwise
        error('unknown task %s', task);
end

txt = jsonencode(sol);
fid = fopen(out_path, 'w');
if fid < 0
    error('cannot write %s', out_path);
end
fwrite(fid, txt, 'char');
fclose(fid);
end
