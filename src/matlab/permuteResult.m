function [newK,newResult, newOrder]=permuteResult(K, result)
%%

fullN=sqrt(size(K,1)); fullM=fullN;
newOrder= randperm(fullN);
% newOrder = 1:fullN
N=fullN; 
M=N;

n=0;
takeEntry=zeros(1,N*M);
for i=1:fullN
    for j=newOrder % only permute second set
%         [i,j]
        ii=sub2ind([fullN,fullM],i,j);
        n=n+1;
        takeEntry(n)=ii;
    end
end

%%
newK=K(takeEntry',takeEntry);
% full(newK);
[~, ass]=getOneHot(result, fullN, fullM);
newAss = ass(newOrder);

linind = sub2ind([fullN,fullM],(1:fullN)',newAss');
newResult = zeros(fullN*fullM,1);
newResult(linind)=1;
end