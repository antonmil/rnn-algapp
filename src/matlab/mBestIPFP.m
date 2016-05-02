function [asgIpfpSMbst, allMarginals] = mBestIPFP(K, Mbst, assOrder)
%%
% Mbest parameters
if nargin<2, Mbst=20; end

param.mbest=Mbst;
param.chck_sols=1;


% [KP, KQ] = conKnlGphPQD(gphs, parKnl);
% K = conKnlGphKD(KP, KQ, gphs);
n = sqrt(size(K,1));
Ct = ones(n);
asgT.X = eye(n);

if nargin<3, assOrder = 1:n; end
asgT.X = asgT.X(assOrder',:);

%% algorithm parameter
[pars, algs] = gmPar(2);
par_mb=pars{6};par_mb{1,3}.alg='ipfp_mbst';
par_mb{1,3}.mbst=param.mbest;
par_mb{1,3}.chck_sols=param.chck_sols;

% tIpfp = tic;
asgIpfpSMbst = gm(K, Ct, asgT, par_mb{:});

% [model, params] = getGurobiModel(n);
% c = ones(1,n*n);
% model.Q = K; c(:)=0; % quadratic weights (c=const means no unaries)
% model.obj = c;
% result = gurobi(model, params); % run gurobi
% 

allMarginals{Mbst} = asgIpfpSMbst.marginals;
if nargout>1
    for m=1:Mbst-1
        par_mb{1,3}.mbst=m;
        asgIpfpSMbstTMP = gm(K, Ct, asgT, par_mb{:});
        allMarginals{m} = asgIpfpSMbstTMP.marginals;
    end
end

% tm(11) = toc(tIpfp);    
% acc(11) = asgIpfpSMbst.acc;
% obj(11) = 0;
% 
% sols{nOut+1,kFs}.IPFPMBST=asgIpfpSMbst.Xmbst(:);
% m_sols{nOut+1,kFs}.IPFPMBST=asgIpfpSMbst.X;
% m_objs{nOut+1,kFs}.IPFPMBST=asgIpfpSMbst.obj;
end