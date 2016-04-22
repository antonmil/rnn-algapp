function DemoMotorMBest_Cluster(Exp)

MaxInstance = 20;
MaxOutliers = 20;
[kFs,nOutP]=ind2sub([MaxInstance,MaxOutliers+1],Exp);
nOut=nOutP-1;
prSet(1);


NofAlgos = 13;
%% src parameter
tag = 'pas';

%% BP parameters
bpoptions.outIter = 1;
bpoptions.innerIter = 10;
BaBoptions.MaxIter = 100;
BaBoptions.bpoptions = bpoptions;

%% Mbest parameters
param.mbest=100;
param.chck_sols=1;


parKnl = st('alg', 'pas'); % type of affinity: only edge distance


acc = zeros(1, NofAlgos);
obj = zeros(1, NofAlgos);
tm = zeros(1, NofAlgos);

%    nOut = 5 ;%-5 ; % randomly remove 2 nodes
parKnl = st('alg', 'pas2'); % type of affinity: only edge distance
%% algorithm parameter
[pars, algs] = gmPar(2);
par_mb=pars{6};par_mb{1,3}.alg='ipfp_mbst';
par_mb{1,3}.mbst=param.mbest;
par_mb{1,3}.chck_sols=param.chck_sols;
%% src
wsSrc = motorAsgSrc(kFs, nOut, 0);
asgT = wsSrc.asgT;

parG = st('link', 'del');
parF = st('smp', 'n', 'nBinT', 4, 'nBinR', 3); % not used, ignore it
wsFeat = motorAsgFeat(wsSrc, parG, parF, 'svL', 1);
[gphs, XPs, Fs] = stFld(wsFeat, 'gphs', 'XPs', 'Fs');

[KP, KQ] = conKnlGphPQD(gphs, parKnl);
K = conKnlGphKD(KP, KQ, gphs);
Ct = ones(size(KP));

%% LSM
tlsm = tic;
asgLsm = gmLSM(K, Ct, asgT, BaBoptions);
tm(1) = toc(tlsm);
acc(1) = asgLsm.acc;
obj(1) = asgLsm.obj;
%% GA
tGa = tic;
asgGa = gm(K, Ct, asgT, pars{1}{:});
tm(2) = toc(tGa);
acc(2) = asgGa.acc;
obj(2) = asgGa.obj;
sols{nOut+1,kFs}.GA=asgGa.X(:);

%% PM
tPm = tic;
asgPm = pm(K, KQ, gphs, asgT);
tm( 3) = toc(tPm);
acc(3) = asgPm.acc;
obj(3) = asgPm.obj;


%% SM
tSm = tic;
asgSm = gm(K, Ct, asgT, pars{3}{:});
tm( 4) = toc(tSm);
acc(4) = asgSm.acc;
obj(4) = asgSm.obj;


%% SMAC
tSmac = tic;
asgSmac = gm(K, Ct, asgT, pars{4}{:});
tm(5) = toc(tSmac);
acc(5) = asgSmac.acc;
obj(5) = asgSmac.obj;

%% IPFP-U
tIpfu = tic;
asgIpfpU = gm(K, Ct, asgT, pars{5}{:});
tm(6) = toc(tIpfu);
acc(6) = asgIpfpU.acc;
obj(6) = asgIpfpU.obj;


%% IPFP-S
tIpfp = tic;
asgIpfpS = gm(K, Ct, asgT, pars{6}{:});
tm( 7) = toc(tIpfp);
acc(7) = asgIpfpS.acc;
obj(7) = asgIpfpS.obj;

%% RRWM
tRrwm = tic;
asgRrwm = gm(K, Ct, asgT, pars{7}{:});
tm( 8) = toc(tRrwm);
acc(8) = asgRrwm.acc;
obj(8) = asgRrwm.obj;

%% FGM-D
tFgmD = tic;
asgFgmD = fgmD(KP, KQ, Ct, gphs, asgT, pars{9}{:});
tm( 9) = toc(tFgmD);
acc(9) = asgFgmD.acc;
obj(9) = asgFgmD.obj;
%% BP MAP solver
tbp = tic;
asgBP = gmBP(K, Ct, asgT,BaBoptions);
tm(10) = toc(tbp);
acc(10) = asgBP.acc;
obj(10) = asgBP.obj;

%% M-Best IPFP-S
tIpfp = tic;
asgIpfpSMbst = gm(K, Ct, asgT, par_mb{:});
tm(11) = toc(tIpfp);
acc(11) = asgIpfpSMbst.acc;
obj(11) = 0;

%% M-Best FGM_D
tFgmD = tic;
asgFgmDMbst = fgmDMBest(KP, KQ, Ct, gphs, asgT, pars{9}{:}, param);
tm( 12) = toc(tFgmD);
acc(12) = asgFgmDMbst.acc;
obj(12) = 0;

%% M-Best BP
tbp = tic;
asgBPMbst = gmMbest(K, Ct, asgT,BaBoptions, param);
tm(13) = toc(tbp);
acc(13) = asgBPMbst.acc;
obj(13) = asgBPMbst.objmbst;


save(['Results/MotorResult_Mbst_Smple',num2str(kFs),'_Outlier',num2str(nOut),'.mat']);
end

