function assign = gmMbest(K, Ct, asgT, options, parm)
[NofNodes, ~] = size(Ct);
[~, NofStates] = size(Ct);
NodeClusters = 0:1:(NofNodes - 1);
NodeClusters = mat2cell(NodeClusters, 1, ones(1, NofNodes));
Npfunc = full(diag(K));
Npfunc2 = reshape(Npfunc, [NofStates, NofNodes]);
NodePotentials2 = mat2cell(Npfunc2, NofStates, ones(1, NofNodes));
EdgesMat = zeros(NofNodes, NofNodes);
K1 = K;

K = K - diag(diag(K));
ecnt2 = 1;
Edges2 = {};

EdgePotentials2 = {};

for i=1:NofNodes
    for j=1:NofNodes
        if(i==j)
            continue;
        end
        efpunc2 = K(((i - 1) * NofStates) + (1:NofStates), ((j - 1) * NofStates) + (1:NofStates) )';
        if(nnz(efpunc2) ~= 0)
            efpunc2 = efpunc2 - 100 * eye(NofStates, NofStates);

            Edges2{ecnt2} = [i - 1, j - 1];
            efpunc2 = full(efpunc2(:));
            EdgePotentials2{ecnt2} = efpunc2(:);
            EdgesMat(i,j) = 1;
            EdgesMat(j, i) = 1;
            ecnt2 = ecnt2 + 1;
        end
    end
end






NodeStates = NofStates * ones(1, NofNodes);
Factors2 = [ NodeClusters, Edges2];%, Triples];


Potentials2 = [NodePotentials2, EdgePotentials2];%, TPFunctions];%, TPfuncs];

NofMBest = parm.mbest;
lambda = 5;


%[decode1, ~, ~, dual1] = MatchingBP(NofNodes, NodeStates, Factors1, Potentials1, options.outIter, options.innerIter,  -1e20);
[y, yvs,tme] = MBestSolverMatching(NofNodes, Factors2, Potentials2, NofMBest, options);
%decode = BaBMatchingSolver(NofNodes, Factors2,Potentials2,options);
%[decode2, ~, ~, dual2] = MatchingBP(NofNodes, NodeStates, Factors2, Potentials2, options.outIter, options.innerIter,  -1e20);

if parm.chck_sols
    if size(unique(y','rows')',2)~=size(y,2)
        error('not unique results')
    end
end


pb = exp((yvs - max(yvs)))/sum(exp(yvs - max(yvs)));


Probabilities = zeros(NofStates, NofStates);
singleAcc = zeros(1, NofMBest);
ObJ=zeros(1,NofMBest);
Xmm=false(NofStates*NofStates,NofMBest);
for mi=1:NofMBest
    cy = y(:,mi);
    Xm=zeros(NofStates,NofStates);
    linearInd = sub2ind([NofStates,NofStates], (1:NofStates)', cy);
    Xm(linearInd)=1;
    ObJ(mi)=Xm(:)' * K1 * Xm(:);
    Xmm(:,mi)=Xm(:);
    cv = exp(-yvs(mi));
    for ni=1:NofNodes
        Probabilities(ni, cy(ni)) = Probabilities(ni, cy(ni)) + pb(mi);
    end
    singleAcc(mi) = matchAsg(Xm , asgT);

end

% cdecode = hungarian(-log(Probabilities));
[~,ixx]=max(Probabilities,[],2);
cdecode=zeros(NofStates,NofStates);
linearInd = sub2ind([NofStates,NofStates], (1:NofStates)', ixx);
cdecode(linearInd)=1;


assign.alg='gmBPMBest';
assign.X = cdecode;
X=assign.X;




acc = matchAsg(assign.X , asgT);

assign.acc = acc;
assign.singleAcc = singleAcc;

assign.Xmbst = X;
assign.objmbst = X(:)' * K1 * X(:);

assign.obj=ObJ;
assign.X = Xmm;

assign.time=tme;

assign.marginals = Probabilities;

end
