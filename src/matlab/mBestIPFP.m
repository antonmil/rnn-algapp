function [asgIpfpSMbst, allMarginals] = mBestIPFP(K, Mbst)
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

%% algorithm parameter
[pars, algs] = gmPar(2);
par_mb=pars{6};par_mb{1,3}.alg='ipfp_mbst';
par_mb{1,3}.mbst=param.mbest;
par_mb{1,3}.chck_sols=param.chck_sols;

% tIpfp = tic;
asgIpfpSMbst = gm(K, Ct, asgT, par_mb{:});

allMarginals{Mbst} = asgIpfpSMbst.marginals;
if nargout>1
    for m=1:Mbst-1
        par_mb{1,3}.mbst=m;
        asgIpfpSMbst = gm(K, Ct, asgT, par_mb{:});
        allMarginals{m} = asgIpfpSMbst.marginals;
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