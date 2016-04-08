function newK=selectSubset(K,N, saveSample)
%%
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

n=0;
takeEntry=zeros(1,N*M);
for i=takePts
    for j=takePts
        ii=sub2ind([fullN,fullM],i,j);
        n=n+1;
        takeEntry(n)=ii;
    end
end
takeEntry=sort(takeEntry);
%%
newK=K(takeEntry',takeEntry);
% full(newK);



if saveSample
    allQ=full(newK);
    save(sprintf('%sdata/test_%d.mat',getRootDir,N),'allQ');
end