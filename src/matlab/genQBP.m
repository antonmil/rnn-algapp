%% quadratic programs
N=3; M=N;
A=zeros(M,N*M); Aeq=zeros(N, N*M);
b=ones(M,1); beq=ones(N,1);
f = zeros(1,N*M);
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
model.obj = f;
model.rhs = [b; beq];
model.vtype = 'B';
model.lb=zeros(1,N*M);
model.ub=ones(1,N*M);

model.modelsense = 'min';
model.sense = char( ['<' * ones(1, length(b)), '=' * ones(1,length(beq))]);
clear params
params.outputflag = 0;    

    
    
mb = 10; % minibatch size
nTrainingSamples = 100;
nSamples = nTrainingSamples * mb;
% nSamples = 1;

allQ = zeros(nSamples, N*M*N*M);
allSol = zeros(nSamples, N*M);

for n=1:nSamples
    if ~mod(n,100), fprintf('.'); end
    Q = randn(N*M, N*M).^5;
    Q=(Q' * Q); % PSD
    Q = Q / max(Q(:));

    model.Q = sparse(Q);

    result = gurobi(model, params);
    allQ(n,:) = Q(:)';
    allSol(n,:) = result.x';
end

%% write out
Qfile = sprintf('../../data/Q_N%d_M%d.txt',N,M);
dlmwrite(Qfile,allQ);
Solfile = sprintf('../../data/Sol_N%d_M%d.txt',N,M);
dlmwrite(Solfile,allSol);
fprintf('\n');