function [ x] = SolveCMRFMatching( NofNodes, NofStates, Factors, Potentials, Phi, gamma0, U, V, MAP, LB, BaBoptions)
%SOLVECMRF Summary of this function goes here
%   Detailed explanation goes here
lsg = -1; rsg = 1;
if(isempty(Phi))
    x = BaBMatchingSolver(NofNodes, NofStates, Factors, Potentials, U, V, BaBoptions);
end

iter = 0;

L = 0; R = gamma0;
gamma = R;
nPotentials = AdditionOfPhiMex(Potentials, Phi, R);
dual = MAP;
nFactors = Factors;
bestv = -1e20;
bestdecode = [];
while(1)
    x1 = BaBMatchingSolver(NofNodes, Factors, nPotentials, U, V, BaBoptions);
    sg = - CluComputeObjMex(x1, NofStates, Factors, Phi);
    rsg = sg;
    if(sg < 0)
        lsg = sg;
        nPotentials = AdditionOfPhiMex(nPotentials, Phi, R);
        L = R;
        R = R * 2;
    else
        v1 = CluComputeObjMex(x1, NofStates, Factors, Potentials);
        sg1 = -CluComputeObjMex(x1, NofStates, Factors, Phi);
        if(sg1 >= 0 && v1 > bestv)
            bestv = v1;
            x = x1;
        end
        break;
    end
end
oldgamma = R;
while(abs(L - R) > 1e-4)
    gamma = (L * (abs(rsg)  + 1) + R * (abs(lsg) +1)) / (abs(rsg) + abs(lsg) +2);
    deltagamma = (gamma - oldgamma);
    oldgamma = gamma;
    nPotentials = AdditionOfPhiMex(nPotentials, Phi, deltagamma);
    [x1] = BaBMatchingSolver(NofNodes, Factors, nPotentials, U, V, BaBoptions);
    cprimal = CluComputeObjMex(x1, NofStates, nFactors, nPotentials);

    sg = - CluComputeObjMex(x1, NofStates, Factors, Phi);
    if(abs(sg) <= 1e-6)
        sg = 1e-6;
    end
    if(sg < 0)
        L = gamma;
        lsg = sg;
    else
        R = gamma;
        rsg = sg;
        v1 = CluComputeObjMex(x1, NofStates, Factors, Potentials);
        if(sg >= 0 && v1 > bestv)
            bestv = v1;
            x = x1;
        end
       
  
    end
    iter = iter + 1;
   % fprintf('iter = %d, dual = %12.7f, primal = %12.7f, IntGap = %12.7f, LRGap = %12.7f, CurrentSg= %f\n', iter, dual, bestv, abs(dual - bestv), abs(L-R), sg);

    if(abs(bestv - MAP) < 1e-6)
        break;
    end
end


end

