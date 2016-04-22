function genQBP(N, mb, nTr, mBst)
%% quadratic program data and solutions

addpath(genpath('.'));
Pair_M = doMatching();

% first set parameters
% problem size
% addpath('Matching');
if nargin<1, N=5; end
if nargin<2, mb = 10; end % minibatch size
if nargin<3, nTr = 10; end % training batches
if nargin<4, mBst=5; end % training batches

doMarginals = mBst>0;
if doMarginals, fprintf('Compute %d-best marginals\n',mBst); end

dataMB = N^4 * mb * nTr * 8 / 1024 / 1024;
if dataMB>1000
    nTr = round(1000/dataMB * nTr);
    fprintf('Too many samples needed. Out-of-memory danger. Reduced to %d\n',nTr)
    dataMB = N^4 * mb * nTr * 8 / 1024 / 1024;
end
fprintf('Data alone will be %.1f MB\n',dataMB);



% N=5;
M=N;
rng('shuffle')
rng(1);


Ns=N;
maxSimThr = 0.8;
sparseFactor = 0.8;

newResEveryNTimes = factorial(N)/2;
newResEveryNTimes = min(newResEveryNTimes, nTr);
% newResEveryNTimes = 1;

fprintf('Gen unique problem every %d samples\n',newResEveryNTimes);

ttmodes = {'train','test'};
% ttmodes = {'train'};

for ttm=ttmodes
    dv = datevec(now);
    dv = [];
    ttmode = char(ttm);
    setTime = tic;
    % ttmode = 'test';
    
    % set up model for gurobi
    clear model params
    [model, params] = getGurobiModel(N);
    
    
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
    data.allMarginals = zeros(nSamples, N*M);
    data.allSolInt = zeros(nSamples, N);
    data.allSolTimes = zeros(nSamples, 1);
    data.optres=false(nSamples,1);
    for m=1:mBst
        allocStr = sprintf('data.all_%d_BestMarginals = zeros(nSamples, N*M);',m);
        eval(allocStr);
    end
    
    n=0;
    while n<nSamples
        if ~mod(n, newResEveryNTimes)
            
            
            %         n
            %     for n=1:nSamples
            
            
            % generate random prob
            %         pot = rand*2;
            %         Q = rand(N*M, N*M).^pot;
            %         if rand<.5
            %             Q = randn(N*M, N*M).^pot;
            %         end
            %
            %         Q=real(Q);
            Q = rand(N*M, N*M);
            
            Q=(Q' * Q); % Positive semi-definite
            
            Q = Q - min(Q(:));    % >= 0
            %         Q(Q>maxSimThr)=maxSimThr;           % max
            Q = Q / max(Q(:));          % <= 1
            %     Q = -Q;                     % negative semi-definite (because we are argmaxing)
            
            
            % sparsify according to real data
            RM = Pair_M{1,randi(length(Pair_M))};
            newK = selectSubset(RM,Ns); %spy(newK);
            Q(~newK)=0;
            %         Q(1:Ns*Ns,1:Ns*Ns) = Q(1:Ns*Ns,1:Ns*Ns).*(~~newK);
            
            
            
            % set diag to rand
            Q(1:N*N+1:end) = rand(1,N*N);
            
            % sparsify more
            rmNPts = randi(N)-1;
            if rmNPts>0
                rmPts = randperm(N); rmPts=rmPts(1:rmNPts);
                rmEntries = zeros(1,length(rmPts)^2);
                nn=0;
                for i=rmPts,
                    for j=rmPts,
                        ii=sub2ind([N,M],i,j);
                        nn=nn+1;
                        rmEntries(nn)=ii;
                        Q(ii,:)=0;
                        Q(:,ii)=0;
                    end
                end
            end
            %         rmPts
            %         rmEntries
            %         pause
            
            Q = Q / max(Q(:));          % <= 1
            
            c = ones(1,N*M);            % linear weights
            
            model.Q = sparse(Q); c(:)=1; % quadratic weights (c=const means no unaries)
            model.obj = c;
            
            if ~isempty(find(isnan(model.Q), 1)), continue; end
            result = gurobi(model, params); % run gurobi
            %         if ~strcmpi(result.status,'optimal')
            %             continue;
            %         end
            
            % pause
            
            % Mbest marginals?
            tocMar = 0;
            if doMarginals
                tic;
                [asgIpfpSMbst, allM] = mBestIPFP(Q,mBst);
  
                tocMar = toc;
            end
            
            
            result.x = binarize(result.x);
            if n<=1
                runtime = tocMar + result.runtime;
                fprintf('Estimated total time: %s. %.3f sec per solution.\n', ...
                    getHMS(runtime* nSamples), runtime);
            end
        else
            %             pause
            %             result.x' * Q * result.x
            [Q, newRes, newOrder] = permuteResult(Q, result.x);
            result.x = newRes;
            if doMarginals, [asgIpfpSMbst, allM] = mBestIPFP(Q,mBst); end
            %             result.x' * Q * result.x
            %             pause
        end
        n=n+1;
        
        nnz = numel(find(Q(:)));
        torchSparse = sparseTensor(Q);
        %         asgIpfpSMbst.marginals
        %         pause
        
        
        % insert in joint matrices
        data.allQ(n,:) = Q(:)'; % WARNING. IS THIS CORRECT INDEXING FOR NON_SYMM MATRICES?
        data.allnnz(n,1) = nnz;
        data.allSparseQ(n,1:nnz*2) = torchSparse(:)';
        data.allc(n,:) = c;
        data.allSol(n,:) = result.x';
        data.allSolTimes(n,1) = result.runtime;
        if doMarginals
            data.allMarginals(n,:) = reshape(asgIpfpSMbst.marginals', 1, N*M);
            for m=1:mBst
                assignStr = sprintf('data.all_%d_BestMarginals(n,:) = reshape(allM{%d}'', 1, N*M);',m,m);
                eval(assignStr);
            end                          
        end
        [~,ass] = getOneHot(result.x);
        data.allSolInt(n,:) = ass;
        if strcmpi(result.status,'optimal')
            data.optres(n,1)=1;
        end
        if ~mod(n,round(nSamples/printDot/10)), fprintf('.'); end
        if ~mod(n,round(nSamples/printDot))
            fprintf('%.2f %%\n',n/nSamples*100);
            %         surf(Q); view(2); colorbar; drawnow;
            %         HeatMap(Q);drawnow;
            
            % write results
            writeQBP(ttmode, N, M, 'QBP', data, n, dv);
        end
        
    end
    
    fprintf('Optimal solutions found: %d / %d (= %.1f %%)\n',numel(find(data.optres)),nSamples,numel(find(data.optres))/nSamples*100);
    fprintf('Avg. runtime per solution: %.2f sec\n',mean(data.allSolTimes));
    fprintf('Total runtime: %s\n',getHMS(toc(setTime)));
    
    
    
end
