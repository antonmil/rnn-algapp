%% quadratic programs
N=3; M=N;
A=zeros(M,N*M); Aeq=zeros(N, N*M);
b=ones(M,1); beq=ones(N,1);
act=1;
for i=1:N
    Aeq(i,act:act+M-1)=1;
    act=act+M;
end
for j=1:M
    A(j,j:M:end)=1;
    act=act+M;
end


clear model
model.A = sparse([A; Aeq]);
model.rhs = [b; beq];
model.vtype = 'B';
model.lb=zeros(1,N*M);
model.ub=ones(1,N*M);
model.modelsense = 'max';
model.sense = char( ['<' * ones(1, length(b)), '=' * ones(1,length(beq))]);
clear params
params.outputflag = 0;

    
    
mb = 10; % minibatch size
nTrainingSamples = 200;
nSamples = nTrainingSamples * mb;
% nSamples = 1;

allQ = zeros(nSamples, N*M*N*M);
allSol = zeros(nSamples, N*M);
allc = zeros(nSamples, N*M);
allSolInt = zeros(nSamples, N);

for n=1:nSamples
    if ~mod(n,100), fprintf('.'); end
    maxPot = 5;
    Q = rand(N*M, N*M).^randi(maxPot);
    if rand<.5
        Q = randn(N*M, N*M).^randi(maxPot);
    end
    Q=(Q' * Q); % PSD
    Q = Q / max(Q(:));
    
    c = rand(1,N*M);

    model.Q = sparse(Q);
    model.obj = c;
    
    result = gurobi(model, params);
    allQ(n,:) = Q(:)'; % WARNING. IS THIS CORRECT INDEXING FOR NON_SYMM MATRICES?
    allc(n,:) = c;    
    allSol(n,:) = result.x';
    [u,v]=find(reshape(result.x,N,M));
    allSolInt(n,:)=u';
end

%% write out
Qfile = sprintf('../../data/Q_N%d_M%d',N,M);
dlmwrite([Qfile,'.txt'],allQ);
save([Qfile,'.mat'],'allQ');

cfile = sprintf('../../data/c_N%d_M%d',N,M);
dlmwrite([cfile,'.txt'],allc);
save([cfile,'.mat'],'allc');


Solfile = sprintf('../../data/Sol_N%d_M%d',N,M);
dlmwrite([Solfile,'.txt'],allSol);
save([Solfile,'.mat'],'allSol');

Solfile = sprintf('../../data/SolInt_N%d_M%d',N,M);
dlmwrite([Solfile,'.txt'],allSolInt);
save([Solfile,'.mat'],'allSolInt');
fprintf('\n');