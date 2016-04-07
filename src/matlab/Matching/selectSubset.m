%%
fullN=size(Ct,1); fullM=fullN;
newOrder= randperm(fullN);
N=4; M=N;
takePts = newOrder(1:N)
n=0;
takeEntry=zeros(1,N*M);
for ii=1:(fullN*fullM)
    [i,j]=ind2sub([fullN,fullM],ii);
    if ismember(i,takePts) && ismember(j,takePts)
        n=n+1;
        takeEntry(n)=ii;
    end
end
takeEntry;
newK=K(takeEntry',takeEntry);
full(newK);
allQ=full(newK);

save('../../../data/test.mat','allQ');