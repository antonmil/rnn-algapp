function [newK,result]=selectSubset(K,N, saveSample, model, params)
%%
result = [];
if nargin<3, saveSample = false; end

fullN=sqrt(size(K,1)); fullM=fullN;
newOrder= randperm(fullN);
% N=4; 
M=N;
takePts = newOrder(1:N);
% takePts = [11 18]
%%
% n=0;
% takeEntry=zeros(1,N*M);
% for ii=1:(fullN*fullM)
%     [i,j]=ind2sub([fullN,fullM],ii);
%     if ismember(i,takePts) && ismember(j,takePts)
%         n=n+1;
%         takeEntry(n)=ii;
%     end
% end
% takeEntry
fullN;
n=0;
takeEntry=zeros(1,N*M);
for i=takePts
    for j=takePts
        ii=sub2ind([fullN,fullM],i,j);
        n=n+1;
        takeEntry(n)=ii;
    end
end
% takeEntry
% t1 = a=meshgrid(takeP,1:3)
% sub2ind([fullN,fullM],takePts,takePts)
% pause
takeEntry=sort(takeEntry);
% pause
%%
newK=K(takeEntry',takeEntry);
% full(newK);



if saveSample
    % save solution as well
    model.Q = newK;
    result = gurobi(model, params);
    result.x = binarize(result.x);
    allSol = result.x';
    [~, allSolInt]=getOneHot(allSol);
    
    asgIpfpSMbst = mBestIPFP(newK,10);
    allMarginals = reshape(asgIpfpSMbst.marginals',1,N*M);
%     [u,~]=find(reshape(result.x,N,M)');

%     allSolInt = u'
%     result.x(:)' * newK * result.x(:)
    
    allQ=full(newK);    
    save(sprintf('%sdata/test_%d.mat',getRootDir,N),'allQ','allSol','allSolInt','allMarginals');
end