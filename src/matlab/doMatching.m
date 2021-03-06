function [Pair_M, allGphs, allFs] = doMatching(dataset)

if nargin<1, dataset='Car'; end

% clear variables;
global footpath;
%delete(gcp)
%parpool(4);
footpath = './Matching';
rng(45678);
addpath(genpath([footpath '/src']));
addpath(genpath([footpath '/lib']));
addpath(genpath([footpath '/STRIPES']));
% addpath('C:\gurobi651\win64\matlab');
% gurobi_setup
prSet(1);

MaxInstance = 30;
if strcmp(dataset,'Motor'), MaxInstance=20; 
else dataset='Car'; MaxInstance=30;
end

NofAlgos = 1;
%% src parameter
tag = 'pas';

%% BP parameters
bpoptions.outIter = 1;
bpoptions.innerIter = 10;
BaBoptions.MaxIter = 50;
BaBoptions.bpoptions = bpoptions;

%% Mbest parameters
param.mbest=5;
param.chck_sols=1;


parKnl = st('alg', 'pas'); % type of affinity: only edge distance
MaxOutliers = 0; % Number of outliers
sols = cell(MaxOutliers+1, MaxInstance);
m_sols = cell(MaxOutliers+1, MaxInstance);
m_objs = cell(MaxOutliers+1, MaxInstance);
marginals = cell(MaxOutliers+1, MaxInstance);
singleAcc = cell(MaxOutliers+1, MaxInstance);

GT = cell(MaxOutliers+1, MaxInstance);
Pair_M = cell(MaxOutliers+1, MaxInstance);
Unary = cell(MaxOutliers+1, MaxInstance);


for nOut = 0:MaxOutliers
    for kFs=1:MaxInstance
        acc = zeros(1, NofAlgos);
        obj = zeros(1, NofAlgos);
        tm = zeros(1, NofAlgos);
        
        parKnl = st('alg', 'pas2'); % type of affinity: only edge distance
        %% algorithm parameter
        [pars, algs] = gmPar(2);
        par_mb=pars{6};par_mb{1,3}.alg='ipfp_mbst';
        par_mb{1,3}.mbst=param.mbest;
        par_mb{1,3}.chck_sols=param.chck_sols;
        %% src
        if strcmp(dataset,'Motor')
            wsSrc = motorAsgSrc(kFs, nOut);
        else
            wsSrc = carAsgSrc(kFs, nOut);
        end
        asgT = wsSrc.asgT;
        
        parG = st('link', 'del'); % Delaunay triangulation for computing the graphs
        parF = st('smp', 'n', 'nBinT', 4, 'nBinR', 3); % not used, ignore it
        wsFeat = motorAsgFeat(wsSrc, parG, parF, 'svL', 1);
        [gphs, XPs, Fs] = stFld(wsFeat, 'gphs', 'XPs', 'Fs');
        
        [KP, KQ] = conKnlGphPQD(gphs, parKnl);
        K = conKnlGphKD(KP, KQ, gphs);
        Ct = ones(size(KP));
        
        GT{nOut+1,kFs}=asgT.X;
        Pair_M{nOut+1,kFs}=K;
        Unary{nOut+1,kFs}=Ct;
        allGphs{nOut+1,kFs}=gphs;
        allFs{nOut+1, kFs} = Fs;
%         size(Ct,1)
        
               %% M-Best BP 
%         tbp = tic;
%         asgBPMbst = gmMbest(K, Ct, asgT,BaBoptions, param);
%         tm(1) = toc(tbp);        
%         acc(1) = asgBPMbst.acc;
%         obj(1) = asgBPMbst.objmbst;
%         sols{nOut+1,kFs}.BPMBST=asgBPMbst.Xmbst(:);
%         m_sols{nOut+1,kFs}.BPMBST=asgBPMbst.X;
%         m_objs{nOut+1,kFs}.BPMBST=asgBPMbst.obj;
%         marginals{nOut+1,kFs}.marginals = asgBPMbst.marginals;
%         singleAcc{nOut+1,kFs}.acc = asgBPMbst.singleAcc;
%         
%         
%         % print information
%         times{nOut+1,kFs} = tm;
%         accs{nOut+1,kFs} = acc;
%         objs{nOut+1,kFs} = obj;
        
    end
end