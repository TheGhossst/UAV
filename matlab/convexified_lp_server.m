function convexified_lp_server(workdir)
% CONVEXIFIED_LP_SERVER  Keep one MATLAB session alive for repeated LP solves.
%
% Protocol (all files under WORKDIR):
%   ready.flag     written after CVX/MOSEK probe succeeds
%   error.txt      written instead of ready.flag on setup failure
%   in.mat         LP payload
%   request.flag   Python writes this after in.mat is complete
%   out.mat        solution
%   response.flag  MATLAB writes this after out.mat is complete
%   stop.flag      Python writes this to shut the server down

    if exist('cvx_begin', 'file') ~= 2
        write_server_error(workdir, 'CVX is not on the MATLAB path. Run cvx_setup first.');
        return;
    end
    try
        cvx_startup_ok();
    catch e
        write_server_error(workdir, getReport(e, 'extended', 'hyperlinks', 'off'));
        return;
    end
    fclose(fopen(fullfile(workdir, 'ready.flag'), 'w'));

    req = fullfile(workdir, 'request.flag');
    stop = fullfile(workdir, 'stop.flag');
    in_mat = fullfile(workdir, 'in.mat');
    out_mat = fullfile(workdir, 'out.mat');
    resp = fullfile(workdir, 'response.flag');

    while true
        if exist(stop, 'file')
            break;
        end
        if exist(req, 'file')
            pause(0.02);
            try
                convexified_lp_file(in_mat, out_mat);
            catch e
                x = [];
                ok = 0;
                st = getReport(e, 'extended', 'hyperlinks', 'off');
                save(out_mat, 'x', 'ok', 'st');
            end
            if exist(req, 'file')
                delete(req);
            end
            fclose(fopen(resp, 'w'));
        else
            pause(0.05);
        end
    end
end


function cvx_startup_ok()
% Tiny LP so a missing MOSEK license fails at session start, not mid-SCA.
    cvx_begin quiet
        cvx_solver mosek
        variable t
        minimize(t)
        subject to
            t >= 0;
    cvx_end
    if ~(strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved'))
        error('CVX+MOSEK probe failed: %s', cvx_status);
    end
end


function write_server_error(workdir, msg)
    fid = fopen(fullfile(workdir, 'error.txt'), 'w');
    fprintf(fid, '%s', msg);
    fclose(fid);
end
