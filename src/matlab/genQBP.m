%% quadratic program data and solutions
% first set parameters
% problem size
% addpath(
N=10;
M=N;
mb = 20; % minibatch size
nTr = 10000; % training batches
maxSimThr = 0.8;
sparseFactor = 0.8;
A=zeros(M,N*M); Aeq=zeros(N, N*M);  % ineq and eq constr. matrices
b=ones(M,1); beq=ones(N,1);         % ineq and eq constr. vectors
act=1;
for i=1:N
    Aeq(i,act:act+M-1)=1;
    act=act+M;
end
for j=1:M
    A(j,j:M:end)=1;
    act=act+M;
end

ttmodes = {'train','test'};
% ttmodes = {'train'};

for ttm=ttmodes
    ttmode = char(ttm);
    setTime = tic;
    % ttmode = 'test';
    
    % set up model for gurobi
    clear model
    model.A = sparse([A; Aeq]);
    model.rhs = [b; beq];
    model.vtype = 'B'; % binary
    % model.vtype = 'C'; model.lb=0*ones(1,N*M); model.ub=1*ones(1,N*M); % relaxed
    model.modelsense = 'max';
    model.sense = char( ['<' * ones(1, length(b)), '=' * ones(1,length(beq))]);
    clear params
    params.outputflag = 0;
    params.TimeLimit = 10; % time limit in seconds
    
    
    %%% How many samples do we want?
    nTrainingSamples = nTr + mb;  % the +mb is for validation set
    nSamples = nTrainingSamples * mb;
    if strcmp(ttmode,'test'), nSamples=10; end
    printDot = 10; % how many times to print
    % nSamples = 1;
    
    
    % for saving all Qs and all solutions
    data=[];
    data.allQ = zeros(nSamples, N*M*N*M);
    data.allSparseQ = zeros(nSamples, N*M*N*M);
    data.allN = zeros(nSamples, 1); % number of non-zeros
    data.allSol = zeros(nSamples, N*M);
    data.allc = zeros(nSamples, N*M);
    data.allSolInt = zeros(nSamples, N);
    data.allSolTimes = zeros(nSamples, 1);
    data.optres=false(nSamples,1);
    
    n=0;
    while n<nSamples
%         n
%     for n=1:nSamples

        
        % generate random prob
        pot = rand*2;
        Q = rand(N*M, N*M).^pot;
        if rand<.5
            Q = randn(N*M, N*M).^pot;
        end
        Q=real(Q);
        
        Q=(Q' * Q); % Positive semi-definite        
        
        Q = Q - min(Q(:));    % >= 0
%         Q(Q>maxSimThr)=maxSimThr;           % max
        Q = Q / max(Q(:));          % <= 1
        %     Q = -Q;                     % negative semi-definite (because we are argmaxing)
           
%         for ii=1:N*M, Q(ii,ii)=0; end % set diag=0
        % only keep immediate neighbors
%         for ii=1:N
%             for jj=1:M
%                 if abs(ii-jj)>1
%                     Qsub = sub2ind([N,N],jj,ii);
%                     Q(Qsub,:)=0;
%                     Q(:,Qsub)=0;
%                 end
%             end
%         end

        % sparsify according to real data
        newK = selectSubset(Pair_M{1,randi(30)},N); %spy(newK);
        Q(~newK)=0;

%         for ii=1:N*M
%             if rand<sparseFactor
%                 Q(ii,:)=0; Q(:,ii)=0;
%             end
%         end

        Q = Q / max(Q(:));          % <= 1
        % set diag to rand
        Q(1:N*N+1:end) = rand(1,N*N);
%         pause

        
        c = rand(1,N*M);            % linear weights
        
        model.Q = sparse(Q); c(:)=0; % quadratic weights (c=0 means no unaries)
        model.obj = c;
        
        if ~isempty(find(isnan(model.Q), 1)), continue; end
        result = gurobi(model, params); % run gurobi
        if ~strcmpi(result.status,'optimal')
            continue;
        end
        n=n+1;
%         result.status = 'TIME_LIMIT';
%         result.x = zeros(N*M,1);result.x(1,1)=1;
        result.x(result.x>0.5)=1;
        result.x(result.x<=0.5)=0;
        if n==1
            fprintf('Estimated total time: %.1f sec (%.1f min / %.1f hr). %.1f sec per solution.\n', ...
                result.runtime * nSamples, result.runtime * nSamples/60, result.runtime * nSamples/3600, result.runtime);
        end
        
        nnz = numel(find(Q(:)));
        torchSparse = sparseTensor(Q);        
        
        % insert in joint matrices
        data.allQ(n,:) = Q(:)'; % WARNING. IS THIS CORRECT INDEXING FOR NON_SYMM MATRICES?        
        data.allnnz(n,1) = nnz;
        data.allSparseQ(n,1:nnz*2) = torchSparse(:)';
        data.allc(n,:) = c;
        data.allSol(n,:) = result.x';        
        data.allSolTimes(n,1) = result.runtime;
        [u,v]=find(reshape(result.x,N,M));
        data.allSolInt(n,:)=u';
        if strcmpi(result.status,'optimal')
            data.optres(n,1)=1;
        end
        if ~mod(n,round(nSamples/printDot/10)), fprintf('.'); end
        if ~mod(n,round(nSamples/printDot))
            fprintf('%.2f %%\n',n/nSamples*100);
            %         surf(Q); view(2); colorbar; drawnow;
            %         HeatMap(Q);drawnow;
            
            % write results
            writeQBP(ttmode, N, M, 'QBP', data, n);
        end        
    end
    
    fprintf('Optimal solutions found: %d / %d (= %.1f %%)\n',numel(find(data.optres)),nSamples,numel(find(data.optres))/nSamples*100);
    fprintf('Avg. runtime per solution: %.2f sec\n',mean(data.allSolTimes));
    fprintf('Total runtime: %.1f sec\n',toc(setTime));
    
    

end