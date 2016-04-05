%% quadratic program data and solutions
% first set parameters
% problem size
N=7;
M=N;
nTr = 100; % training batches
maxSimThr = 0.8;
sparseFactor = 0.2;
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

for ttm=ttmodes
    ttmode = char(ttm);
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
    params.TimeLimit = .1; % time limit in seconds
    
    
    %%% How many samples do we want?
    mb = 10; % minibatch size
    nTrainingSamples = nTr + mb;  % the +mb is for validation set
    nSamples = nTrainingSamples * mb;
    if strcmp(ttmode,'test'), nSamples=10; end
    printDot = 10; % how many times to print
    % nSamples = 1;
    
    
    % for saving all Qs and all solutions
    allQ = zeros(nSamples, N*M*N*M);
    allSol = zeros(nSamples, N*M);
    allc = zeros(nSamples, N*M);
    allSolInt = zeros(nSamples, N);
    optres=false(nSamples,1);
    
    for n=1:nSamples
        if ~mod(n,round(nSamples/printDot))
            fprintf('%.2f %%\n',n/nSamples*100);
            
            %         surf(Q); view(2); colorbar; drawnow;
            %         HeatMap(Q);drawnow;
        end
        
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
           
        for ii=1:N*M, Q(ii,ii)=0; end % set diag=0
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

        for ii=1:N*M
            if rand<sparseFactor
                Q(ii,:)=0; Q(:,ii)=0;
            end
        end
        Q = Q / max(Q(:));          % <= 1        

        
        c = rand(1,N*M);            % linear weights
        
        model.Q = sparse(Q); c(:)=0; % quadratic weights (c=0 means no unaries)
        model.obj = c;
        
        result = gurobi(model, params); % run gurobi
        result.x(result.x>0.5)=1;
        result.x(result.x<=0.5)=0;
        
        % insert in joint matrices
        allQ(n,:) = Q(:)'; % WARNING. IS THIS CORRECT INDEXING FOR NON_SYMM MATRICES?
        allc(n,:) = c;
        allSol(n,:) = result.x';        
        [u,v]=find(reshape(result.x,N,M));
        allSolInt(n,:)=u';
        if strcmpi(result.status,'optimal')
            optres(n,1)=1;
        end
    end
    
    fprintf('Optimal solutions found: %d / %d (= %.1f %%)\n',numel(find(optres)),nSamples,numel(find(optres))/nSamples*100);
    
    
    %% write out
    Qfile = sprintf('../../data/%s/Q_N%d_M%d',ttmode,N,M);
    dlmwrite([Qfile,'.txt'],allQ);
    save([Qfile,'.mat'],'allQ');
    
    cfile = sprintf('../../data/%s/c_N%d_M%d',ttmode,N,M);
    dlmwrite([cfile,'.txt'],allc);
    save([cfile,'.mat'],'allc');
    
    
    Solfile = sprintf('../../data/%s/Sol_N%d_M%d',ttmode,N,M);
    dlmwrite([Solfile,'.txt'],allSol);
    save([Solfile,'.mat'],'allSol');
    
    Solfile = sprintf('../../data/%s/SolInt_N%d_M%d',ttmode,N,M);
    dlmwrite([Solfile,'.txt'],allSolInt);
    save([Solfile,'.mat'],'allSolInt');
    fprintf('\n');
end