function [ solutions, objs, time_stats ] = MBestSolverMatching( NofNodes, Factors, Potentials, M , BaBOptions)
addpath('../STRIPES/');
NofStates = ones(1, NofNodes) * NofNodes;
%MBESTSOLVER Summary of this function goes here
%   Detailed explanation goes here
x = zeros(NofNodes, M);
y = zeros(NofNodes, M);
xv = zeros(M,1);
yv = zeros(M,1);
time_stats = zeros(M,1);
Constraints = cell(M,1);
NodePositions = - ones(NofNodes, 1);
U=zeros(1,NofNodes);
V = zeros(1, NofNodes);
tic
decode = BaBMatchingSolver(NofNodes, Factors, Potentials, zeros(1,NofNodes),  zeros(1, NofNodes), BaBOptions);

toc
nFactors = Factors;
nPotentials = Potentials;
        
EdgesMat = sparse(NofNodes, NofNodes);
 ne = length(Factors);
for i=1:ne
    if(length(nFactors{i}) == 2)
        ii = nFactors{i}(1) + 1;
        jj = nFactors{i}(2) + 1;
        EdgesMat(ii, jj ) = i;
        EdgesMat(jj, ii ) = i;
    end
end
[tree_Edges, tree_pair_inds, tree_weight, tree_num_conn_comp] = ...
    calcMaxSpanningTree(decode', EdgesMat, [], [], []);
DegreeNodes = zeros(NofNodes,1);
for i=1:ne
    if(length(nFactors{i}) == 2)
        ii = nFactors{i}(1) + 1;
        jj = nFactors{i}(2) + 1;
        if(tree_pair_inds(ii, jj) == i)
            DegreeNodes(ii) = DegreeNodes(ii) + 1;
            DegreeNodes(jj) = DegreeNodes(jj) + 1;
        end
    end
end

gamma0 = 0.1;


tstart = tic;

for m=1:M
    if(m==1)
        gamma1 = gamma0;
       
        x(:,m) = decode + 1;
        xv(m) = CluComputeObjMex(decode, NofStates, Factors, Potentials);
    else
        [c, k] = max(yv(1:(m-1)));
        x(:,m) = y(:,k);
        xv(m) = yv(k);
        fprintf('The %d-Best Solution is: %20.8f\n', m, xv(m));
        diff1 = (x(:,m) ~= x(:,k));
        diff = find(diff1==1);
        if(~isempty(diff))
            Constraints{m} = [Constraints{k}; diff(1), x(diff(1), m)];
            Constraints{k} = [Constraints{k}; diff(1), -x(diff(1), m)];
        end
        [y1] =  CalcNextBestSolution(Constraints{k}, x(:,k),NofNodes, NofStates, nFactors,nPotentials, xv(k), DegreeNodes, tree_pair_inds,U,V, gamma0,BaBOptions);
        yv(k) = CluComputeObjMex(y1, NofStates, Factors, Potentials);
        y(:,k) = y1 + 1;
    
    end
    [ y2] =  CalcNextBestSolution(Constraints{m}, x(:,m),NofNodes, NofStates, nFactors,nPotentials, xv(m), DegreeNodes, tree_pair_inds,U,V, gamma0,BaBOptions);   
    yv(m) = CluComputeObjMex(y2, NofStates, Factors, Potentials);
    y(:,m) = y2 + 1;
    time_stats(m) = toc(tstart);
end
solutions = x;
objs = xv;


end

function [y] = CalcNextBestSolution(Constraints, xstar,NofNodes, NofStates, Factors,Potentials, bv, DegreeNodes, tree_pair_inds, U, V, gamma0, BaBOptions)
lFactors = length(Factors);
Phi = arrayfun(@(x) GenPhi(NofStates, tree_pair_inds, Factors, x, DegreeNodes, xstar - 1), 1:length(Factors), 'UniformOutput', 0);
[NofConstraints, ~] = size(Constraints);
for i=1:NofConstraints
    Idx  = Constraints(i, 1) ;
    assign = Constraints(i, 2);
    if(assign > 0)
        for i=1:NofNodes
            if(i ~= assign)
                Potentials{Idx}(i) = -100;
            end
        end
    else
        assign = -assign;
        Potentials{Idx}(assign) = -100;
    end
end

tic
[y] = SolveCMRFMatching( NofNodes, NofStates, Factors, Potentials, Phi, gamma0, U, V, bv, -1e20, BaBOptions);
toc

end

function phi = GenPhi(StatesOfNodes, tree_pair_inds, Factors, i, DegreeNodes, decode)
phi = [];
if(length(Factors{i}) == 1)
    nodeidx = Factors{i} + 1;
    if(DegreeNodes(nodeidx) ~= 1)
        phi = zeros(StatesOfNodes(nodeidx), 1);
        phi(decode(nodeidx) + 1) = 1 - DegreeNodes(nodeidx);
    end
elseif(length(Factors{i}) == 2)
    ii = Factors{i}(1) + 1;
    jj = Factors{i}(2) + 1;
    if(tree_pair_inds(ii, jj) == i)
        phi = zeros(StatesOfNodes(jj), StatesOfNodes(ii));
        phi(decode(jj) + 1, decode(ii) + 1) = 1;
        phi = reshape(phi, [StatesOfNodes(ii) * StatesOfNodes(jj), 1 ]);
    end
end
end