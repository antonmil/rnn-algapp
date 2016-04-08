function writeQBP(ttmode, N, M, fname, data,n)
    %% write out
    allQ = data.allQ(1:n,:);
    allSparseQ = data.allSparseQ(1:n,:);
    allnnz = data.allnnz(1:n,:);
    allc = data.allc(1:n,:);
    allSol = data.allSol(1:n,:);
    allSolInt = data.allSolInt(1:n,:);
    
    filename = sprintf('../../data/%s/%s_N%d_M%d',ttmode,fname,N,M);
%     dlmwrite([filename,'.txt'],data);
    save([filename,'.mat'],'allQ','allSparseQ','allnnz','allc','allSol','allSolInt','-v7.3');
    
%     fprintf('\n');
end