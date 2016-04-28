%%
addpath(genpath('.'))
nRuns = 10;

rng(321);
if ~exist('Pair_M','var')
    Pair_M=doMatching();
end

N=5;
rnnSize = 32;
numLayers = 1;
solIndex = 1; % 1=integer, 2=distribution
infIndex = 1; % 1=map, 2=marginal
[gurModel, gurParams] = getGurobiModel(N);
model_sign = sprintf('mt1_r%d_l%d_n%d_m%d_o2_s%d_i%d_valen',rnnSize, numLayers, N,N, solIndex, infIndex);
model_name = 'trainHun';
% model_name = '0428Bg-4';
mBst = 10;
doRandomize = true;
% doRandomize = false;

runInfos.allAcc=zeros(1,nRuns);
runInfos.gurAcc=zeros(1,nRuns);
runInfos.mbstAcc=zeros(1,nRuns);
runInfos.mbstHAAcc=zeros(1,nRuns);
runInfos.IPFPAcc=zeros(1,nRuns);
runInfos.moptAcc=zeros(1,nRuns);
runInfos.rnnHunAcc=zeros(1,nRuns);

runInfos.allObj=zeros(1,nRuns);
runInfos.gurObj=zeros(1,nRuns);
runInfos.mbstObj=zeros(1,nRuns);
runInfos.mbstHAObj=zeros(1,nRuns);
runInfos.IPFPObj=zeros(1,nRuns);
runInfos.moptObj=zeros(1,nRuns);
runInfos.rnnHunObj=zeros(1,nRuns);

runInfos.rnnTime=zeros(1,nRuns);
runInfos.gurTime=zeros(1,nRuns);
runInfos.mbstTime=zeros(1,nRuns);
runInfos.mbstHAObj=zeros(1,nRuns);
runInfos.IPFPTime=zeros(1,nRuns);
runInfos.moptTime=zeros(1,nRuns);
runInfos.rnnHunTime=zeros(1,nRuns);

asgT.X=eye(N);

for r = 1:nRuns
    fprintf('.');
%     rng(321);
    RM = Pair_M{1,randi(length(Pair_M))};    
    [newK,gurResult] = selectSubset(RM, N, true, gurModel, gurParams);

    runInfos.gurTime(r) = gurResult.runtime;
    [gurMat, gurAss] = getOneHot(gurResult.x);

    % randomize
    newAss = 1:N;
    if doRandomize
        canSol = eye(N);
        [newK, newSol, newOrder]=permuteResult(newK, canSol(:)');
        [newSolMat, newAss] = getOneHot(newSol);
        asgT.X = eye(N);   
        asgT.X = asgT.X(newAss',:);
        gurModel.Q = newK;
        gurResult = gurobi(gurModel, gurParams);
        runInfos.gurTime(r) = gurResult.runtime;    
        [gurMat, gurAss] = getOneHot(gurResult.x);
    end
    
    if length(unique(gurAss)) ~= length(gurAss)
%         fprintf('Gurobi solution not one-to-one!\n')
    end
    
    % marg
    asgIpfpSMbst = mBestIPFP(newK,mBst,newAss);
    mbstVec = reshape(asgIpfpSMbst.Xmbst,N*N,1);
    runInfos.mbstTime(r) = sum(asgIpfpSMbst.time);
    runInfos.mbstObj(r) = mbstVec' * newK * mbstVec;
    runInfos.mbstAcc(r) = matchAsg(asgIpfpSMbst.Xmbst', asgT);
    
    thun = tic;
    [matchHun, costHun] = hungarian(-asgIpfpSMbst.marginals);
    thun=toc(thun);
    hunVec = reshape(matchHun,N*N,1);
    runInfos.mbstHATime(r) = runInfos.mbstTime(r) + thun;
    runInfos.mbstHAObj(r) = hunVec' * newK * hunVec;
    runInfos.mbstHAAcc(r) = matchAsg(matchHun', asgT);
    
    % IPFP 'best'
    [pars, algs] = gmPar(2);
    
    Ct = ones(sqrt(size(newK,1)));    
    asgIpfpS = gm(newK, Ct, asgT, pars{6}{:});

    IPFPVec = reshape(asgIpfpS.X,N*N,1);
    runInfos.IPFPTime(r) = asgIpfpS.tim;
    runInfos.IPFPObj(r) = IPFPVec' * newK * IPFPVec;
    runInfos.IPFPAcc(r) = matchAsg(asgIpfpS.X', asgT);

    % IPFP 'opt-out-of-m'
    [~,m] = max(asgIpfpSMbst.obj);
    moptVec = reshape(asgIpfpSMbst.X(:,:,m),N*N,1);
    runInfos.moptTime(r) = sum(asgIpfpSMbst.time);
    runInfos.moptObj(r) = moptVec' * newK * moptVec;
    runInfos.moptAcc(r) = matchAsg(asgIpfpSMbst.X(:,:,m)', asgT);
    
    try
    cmd = sprintf('cd ..; pwd; th %s.lua -model_name %s -model_sign %s -suppress_x 1','test', model_name , model_sign);
    [a,b] = system(cmd);    
    if a~=0
%         fprintf('Error running RNN!\n'); b
%         break;
    end

    resRaw = dlmread(sprintf('../../out/%s_%s.txt',model_name, model_sign));
    runInfos.rnnTime(r) = resRaw(1,3);
%     resRaw(:,1) = reshape(reshape(resRaw(:,1),N,N)',N*N,1);
    resVec = resRaw(:,1);
    [myResMat, myAss] = getOneHot(resVec);
%     myAss
    if length(unique(myAss)) ~= length(myAss)
%         fprintf('RNN solution not one-to-one!\n')
    end
    
    obj = resVec(:)' * newK * resVec(:);
    acc = matchAsg(myResMat, asgT);
%     fprintf('RNN Accuracy: %.2f\n',acc)        
    runInfos.allAcc(r)=acc;
    runInfos.allObj(r) = obj;
    
    % resolve with hungarian
    resObj = resRaw(:,2);
    resMat = reshape(resObj,N,N)';
    thun = tic;
    [matchHun, costHun] = hungarian(-resMat);
    thun=toc(thun);
    hunVec = reshape(matchHun',N*N,1);
    runInfos.rnnHunTime(r) = runInfos.rnnTime(r) + thun;
    runInfos.rnnHunObj(r) = hunVec' * newK * hunVec;
    runInfos.rnnHunAcc(r) = matchAsg(matchHun, asgT);    
%     pause
    catch err
        fprintf('WARNING. LSTM IGNORED. %s\n',err.message);
    end
    
    acc = matchAsg(gurMat, asgT);
    obj = gurResult.x(:)' * newK * gurResult.x(:);
%     fprintf('Gur Accuracy: %.2f\n\n',acc)
    
    runInfos.gurAcc(r)=acc;
    runInfos.gurObj(r) = obj;
end

% fprintf('Average RNN Accuracy: %.2f\n',mean(runInfos.allAcc))
% fprintf('Average Gur Accuracy: %.2f\n',mean(runInfos.gurAcc))

%% Stats
mbstMethod = sprintf('%d-bstMar',mBst);
mbstHAMethod = sprintf('%d-bstMarH',mBst);
moptMethod = sprintf('%d-opt',mBst);
fprintf('\n%10s|%8s|%8s|%8s\n','Method','acc','obj','time');
fprintf('-------------------------------------\n');
fprintf('%10s|%8.2f|%8.2f|%8.3f\n','IPFP',mean(runInfos.IPFPAcc),mean(runInfos.IPFPObj),mean(runInfos.IPFPTime));
fprintf('%10s|%8.2f|%8.2f|%8.3f\n','Gurobi',mean(runInfos.gurAcc),mean(runInfos.gurObj),mean(runInfos.gurTime));
fprintf('%10s|%8.2f|%8.2f|%8.3f\n','LSTM',mean(runInfos.allAcc),mean(runInfos.allObj),mean(runInfos.rnnTime));
fprintf('%10s|%8.2f|%8.2f|%8.3f\n','LSTM-HUN',mean(runInfos.rnnHunAcc),mean(runInfos.rnnHunObj),mean(runInfos.rnnHunTime));
fprintf('%10s|%8.2f|%8.2f|%8.3f\n',mbstMethod,mean(runInfos.mbstAcc),mean(runInfos.mbstObj),mean(runInfos.mbstTime));
fprintf('%10s|%8.2f|%8.2f|%8.3f\n',mbstHAMethod,mean(runInfos.mbstHAAcc),mean(runInfos.mbstHAObj),mean(runInfos.mbstHATime));
fprintf('%10s|%8.2f|%8.2f|%8.3f\n',moptMethod,mean(runInfos.moptAcc),mean(runInfos.moptObj),mean(runInfos.moptTime));
