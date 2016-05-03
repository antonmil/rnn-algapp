function writeQBP(ttmode, N, M, fname, data,n,dv)
    %% write out
    allQ = data.allQ(1:n,:);
    allSparseQ = data.allSparseQ(1:n,:);
    allnnz = data.allnnz(1:n,:);
    allc = data.allc(1:n,:);
    allSol = data.allSol(1:n,:);
    allSolInt = data.allSolInt(1:n,:);
    allMarginals = data.allMarginals(1:n,:);
    
    % look for all_m_BestMarginal fields
    fields = fieldnames(data);
    for f=fields'
        cf = char(f);
        if strfind(cf,'all_')
            assignStr = sprintf('%s = data.%s(1:n, :);',cf,cf);
            eval(assignStr);
        end
    end
    
    if nargin<7, dv = datevec(now); end
    
    if isempty(dv), filename = sprintf('../../data/%s/%s_N%d_M%d',ttmode,fname,N,M); 
    else
        timestamp = sprintf('%02d%02d-%02d%02d%02d',dv(2),dv(3),dv(4),dv(5),round(dv(6)));
        filename = sprintf('../../data/%s/%s_N%d_M%d-%s',ttmode,fname,N,M,timestamp);
    end
    
    
%     dlmwrite([filename,'.txt'],data);
    save([filename,'.mat'],'all*','-v7.3');
    
%     fprintf('\n');
end