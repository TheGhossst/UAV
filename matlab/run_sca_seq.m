function run_sca_seq(in_path, out_path)
%RUN_SCA_SEQ  Batch entry for sequential SCA in MATLAB CVX/MOSEK.
% One MATLAB session on purpose: CVX startup is ~60 s.

setup_cvx_mosek();
raw = fileread(in_path);
in = jsondecode(raw);
out = sca_seq(in);
txt = jsonencode(out);
fid = fopen(out_path, 'w');
if fid < 0
    error('cannot write %s', out_path);
end
fwrite(fid, txt, 'char');
fclose(fid);
end
