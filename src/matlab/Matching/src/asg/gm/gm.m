function asg = gm(K, Ct, asgT, parIni, parPosC, parPosD)
% Graph matching.
%
% This function can be used as the interface of the following algorithms:
%   Graudate Assignment (GA)
%   Spectral Matching (SM)
%   Spectral Matching with Affine Constraint (SMAC)
%   Integer Projected Fixed Point (IPFP)
%   Reweighted Random Walks Matching (RRWM)
%
% Math
%   This code is to solve the following problem:
%     max_X   vec(X)' * K * vec(X)
%     s.t.    X is a permutation matrix
%
% Remark
%   nn = n1 x n2
%
% Input
%   K        -  affinity matrix, nn x nn (sparse)
%   Ct       -  correspondence constraint, n1 x n2
%                 Ct_ij = 1: i and j can be matched
%                 Ct_ij = 0: i and j cannot be matched
%   asgT     -  ground-truth assignment (can be [])
%   parIni   -  parameter for initialization
%   parPosC  -  parameter for continuous-continuous post-processing
%   parPosD  -  parameter for continuous-discrete post-processing
%
% Output
%   asg      -  assignment
%     alg    -  algorithm name
%     X      -  binary correspondence matrix, n1 x n2
%     acc    -  accuracy (= 0 if asgT is [])
%     obj    -  objective value
%     tim    -  time cost
%
% History
%   create   -  Feng Zhou (zhfe99@gmail.com), 01-25-2009
%   modify   -  Seyed Hamid Rezatofighi(s.h.r.tofighi@gmail.com), 22-10-2015

% function parameter
prIn('gm', 'ini %s, posC %s, posD %s', parIni.alg, parPosC.alg, parPosD.alg);
ha = tic;

% initialization
X0 = gmIni(K, Ct, parIni);

% continous -> continous
XC = gmPosC(K, Ct, X0, parPosC);

% continous -> discrete

if strcmp(parPosD.alg,'ipfp_mbst')
    [Xm,objm,tme] = gmPosD_m(K, Ct, XC, parPosD);
    
    xvs = objm - min(objm);
    xvs = full(exp(xvs));
    xvs = xvs / sum(xvs);
    
    Probability = zeros(size(Xm,1),size(Xm,2));
    for i=1:parPosD.mbst
        Probability = Probability+Xm(:,:,i).* xvs(i);
    end
    
    [~, idxs] = max(Probability,[],2);
    ridxs = sub2ind(size(Probability), (1:size(Probability,1))', idxs);
    X = zeros(size(Probability));
    X(ridxs) = 1;
    % matching with ground-truth assignment if possible
    acc = matchAsg(X, asgT);
    
    % store
    asg.alg = 'ipfpMBest';
    asg.obj = objm;
    asg.acc = acc;
    
    asg.Xmbst = X;
    asg.X = Xm;
    asg.time=tme;
    
    prOut;
else
    X = gmPosD(K, Ct, XC, parPosD);
    % compare with ground-truth
    acc = matchAsg(X, asgT);
    
    % store
    asg.alg = sprintf('gm+%s+%s+%s', parIni.alg, parPosC.alg, parPosD.alg);
    asg.X = X;
    asg.acc = acc;
    asg.obj = X(:)' * K * X(:);
    asg.tim = toc(ha);
    
    prOut;
    
end



