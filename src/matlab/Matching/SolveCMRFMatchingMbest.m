function [ x] = SolveCMRFMatchingMbest( NofNodes, NofStates, Factors, Potentials, Phis, U, V, MAP, BaBoptions)
%SOLVECMRF Summary of this function goes here
%   Detailed explanation goes here
addpath('../IsingModel/');
if(isempty(Phis))
    x = BaBMatchingSolver(NofNodes,  Factors, Potentials, U, V, BaBoptions);
end


nOfConstraints = length(Phis);

nPotentials = Potentials;
dual = MAP;
nFactors = Factors;
bestv = -1e20;
bestdecode = [];

for i=1:300
    isFeasible = 1;
    x1 = BaBMatchingSolver(NofNodes,  Factors, nPotentials, U, V, BaBoptions);
    
    sg = zeros(nOfConstraints, 1);
    for ci = 1:nOfConstraints
        sg(ci) = -CluComputeObjMex(x1, NofStates, Factors, Phis{ci});
        if(sg(ci) < 0)
            isFeasible = 0;
        end
    end
    sg = sg / norm(sg);
    sg = sg * (1 / i);
    for ci = 1:nOfConstraints
        nPotentials = AdditionOfPhiMex(nPotentials, Phis{ci}, -sg(ci));
    end
    if(isFeasible == 1)
        cv = CluComputeObjMex(x1, NofStates, Factors, Potentials);
        if(cv > bestv)
            bestv = cv;
            bestdecode = x1;
        end
    end
end

x = bestdecode;
end

