load('codeInfo.mat')
jsonStr = jsonencode(codeInfo);
fid = fopen('codeInfo.json', 'w');
if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, jsonStr, 'char');
fclose(fid);