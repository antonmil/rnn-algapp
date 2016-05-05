%%
addpath(genpath('.'))
nRuns = 1:10;


rng('shuffle');
rng(321);
if ~exist('Pair_M','var')
    [Pair_M, allGphs, allFs]=doMatching('Motor');
end

N=8;
rnnSize = 64;
numLayers = 2;
solIndex = 1; % 1=integer, 2=distribution
infIndex = 1; % 1=map, 2=marginal
[gurModel, gurParams] = getGurobiModel(N);
model_sign = sprintf('mt1_r%d_l%d_n%d_m%d_o2_s%d_i%d_valen',rnnSize, numLayers, N,N, solIndex, infIndex);
model_name = 'trainHun';
model_name = '0502Fs-2'; % GOOD ONE (also 0502Fs-1)
model_name = '0505As-2'; %

mBst = 10;
doRandomize = true;
% doRandomize = false;

allRes = {}; mInd = 0;


fprintf('Testing %s - %s\n', model_name, model_sign);
asgT.X=eye(N);

allGphsSubSel = allGphs;
allFs = allFs;

for r = nRuns
    fprintf('.');
    %     rng(3211211);
    randSmpl = randi(length(Pair_M));
    RM = Pair_M{1,randSmpl};
    [newK,gurResult,takePts] = selectSubset(RM, N, false, gurModel, gurParams);
    
    for ii=1:2
        allGphsSubSel{randSmpl}{ii}.Pt=allGphs{randSmpl}{ii}.Pt(:,takePts);
        allGphsSubSel{randSmpl}{ii}.vis=allGphs{randSmpl}{ii}.vis(takePts',takePts);
        allGphsSubSel{randSmpl}{ii}.G=allGphs{randSmpl}{ii}.G(takePts,:);
        allGphsSubSel{randSmpl}{ii}.H=allGphs{randSmpl}{ii}.H(takePts,:);
        allGphsSubSel{randSmpl}{ii}.XP=allGphs{randSmpl}{ii}.XP(1,takePts);
    end
    
    
    %     runInfos.gurTime(r) = gurResult.runtime;
    %     [gurMat, gurAss] = getOneHot(gurResult.x);
    
    % randomize
    GTAss = 1:N;
    if doRandomize
        canSol = eye(N);
        [newK, newSol, newOrder]=permuteResult(newK, canSol(:)');
        [newSolMat, GTAss] = getOneHot(newSol);
        asgT.X = eye(N);
        asgT.X = asgT.X(GTAss',:);
        %         asgT.X = asgT.X(:,GTAss);
    end
    %     newK(~~newK) = rand;
    
    gurModel.Q = newK;
    gurResult = gurobi(gurModel, gurParams);
    %     gurResult
    runInfos.gurTime(r) = gurResult.runtime;
    [gurMat, gurAss] = getOneHot(gurResult.x);
    acc = matchAsg(gurMat, asgT);
    obj = gurResult.x(:)' * newK * gurResult.x(:);
    %     fprintf('Gur Accuracy: %.2f\n\n',acc)
    
    %     runInfos.gurAcc(r)=acc;
    %     runInfos.gurObj(r) = obj;
    %%% GUROBI
    mInd=1;
    allRes{mInd}.name = 'Branch-and-cut';
    allRes{mInd}.cite = 'gurobi';
    allRes{mInd}.acc(r) = acc;
    allRes{mInd}.obj(r) = obj;
    allRes{mInd}.time(r) = gurResult.runtime;
    allRes{mInd}.optimal(r) = strcmpi(gurResult.status,'OPTIMAL');
    allRes{mInd}.resMat(:,:,r) = gurMat;
    
    %     asgT.X = gurMat;
    
    
    if length(unique(gurAss)) ~= length(gurAss)
        %         fprintf('Gurobi solution not one-to-one!\n')
    end
    
    
    
    
    % IPFP-S
    [pars, algs] = gmPar(2);
    
    Ct = ones(sqrt(size(newK,1)));
    asgIpfpS = gm(newK, Ct, asgT, pars{6}{:});
    IPFPVec = reshape(asgIpfpS.X,N*N,1);
    
    mInd=mInd+1;
    allRes{mInd}.name = sprintf('IPFP-S');
    allRes{mInd}.cite = 'Leordeanu:2012:IJCV';
    allRes{mInd}.acc(r) = matchAsg(asgIpfpS.X', asgT);
    allRes{mInd}.obj(r) = IPFPVec' * newK * IPFPVec;
    allRes{mInd}.time(r) = asgIpfpS.tim;
    allRes{mInd}.resMat(:,:,r) = asgIpfpS.X';
    
    
    %     runInfos.IPFPTime(r) = asgIpfpS.tim;
    %     runInfos.IPFPObj(r) = IPFPVec' * newK * IPFPVec;
    %     runInfos.IPFPAcc(r) = matchAsg(asgIpfpS.X', asgT);
    
    % IPFP 'opt-out-of-m'
    asgIpfpSMbst = mBestIPFP(newK,mBst,GTAss);
    [~,m] = max(asgIpfpSMbst.obj);
    moptVec = reshape(asgIpfpSMbst.X(:,:,m),N*N,1);
    
    mInd=mInd+1;
    allRes{mInd}.name = sprintf('IPFP-%dbstOpt',mBst);
    allRes{mInd}.acc(r) = matchAsg(asgIpfpSMbst.X(:,:,m)', asgT);
    allRes{mInd}.obj(r) = moptVec' * newK * moptVec;
    allRes{mInd}.time(r) = sum(asgIpfpSMbst.time);
    allRes{mInd}.resMat(:,:,r) = asgIpfpSMbst.X(:,:,m)';
    
    %     runInfos.moptTime(r) = sum(asgIpfpSMbst.time);
    %     runInfos.moptObj(r) = moptVec' * newK * moptVec;
    %     runInfos.moptAcc(r) = matchAsg(asgIpfpSMbst.X(:,:,m)', asgT);
    
    
    % marginals
    mbstVec = reshape(asgIpfpSMbst.Xmbst,N*N,1);
    %     runInfos.mbstTime(r) = sum(asgIpfpSMbst.time);
    %     runInfos.mbstObj(r) = mbstVec' * newK * mbstVec;
    %     runInfos.mbstAcc(r) = matchAsg(asgIpfpSMbst.Xmbst', asgT);
    
    mInd=mInd+1;
    allRes{mInd}.name = sprintf('IPFP-%dbstMar',mBst);
    allRes{mInd}.cite = 'Rezatofighi:2016:CVPR';
    allRes{mInd}.acc(r) = matchAsg(asgIpfpSMbst.Xmbst', asgT);
    allRes{mInd}.obj(r) = mbstVec' * newK * mbstVec;
    allRes{mInd}.time(r) = sum(asgIpfpSMbst.time);
    allRes{mInd}.resMat(:,:,r) = asgIpfpSMbst.Xmbst';
    
    % marginals with Hungarian
    thun = tic;
    [matchHun, costHun] = hungarian(-asgIpfpSMbst.marginals);
    thun=toc(thun);
    hunVec = reshape(matchHun,N*N,1);
    %     runInfos.mbstHATime(r) = runInfos.mbstTime(r) + thun;
    %     runInfos.mbstHAObj(r) = hunVec' * newK * hunVec;
    %     runInfos.mbstHAAcc(r) = matchAsg(matchHun', asgT);
    mInd=mInd+1;
    allRes{mInd}.name = sprintf('IPFP-%dbstMarHA',mBst);
    allRes{mInd}.acc(r) = matchAsg(matchHun', asgT);
    allRes{mInd}.obj(r) = hunVec' * newK * hunVec;
    allRes{mInd}.time(r) = allRes{mInd-1}.time(r) + thun;
    allRes{mInd}.resMat(:,:,r) = matchHun';
    
    % LSTM
    allQ=full(newK); allQ=allQ(:)';
    gurResult.x = binarize(gurResult.x);
    allSol = reshape(asgT.X', 1, N*N);
    allSolInt = GTAss;
    allMarginals = reshape(asgIpfpSMbst.marginals',1,N*N);
    testfilebase='test';
    testfile = sprintf('%sdata/%s_%d.mat',getRootDir,testfilebase,N);
    save(testfile,'allQ','allSol','allSolInt','allMarginals');
    
    try
        cmd = sprintf('cd ..; pwd; th %s.lua -model_name %s -model_sign %s -suppress_x 1 -test_file %s','test', ...
            model_name , model_sign, testfilebase);
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
        
        %     obj = resVec(:)' * newK * resVec(:);
        %     acc = matchAsg(myResMat, asgT);
        %     fprintf('RNN Accuracy: %.2f\n',acc)
        %     runInfos.allAcc(r)=acc;
        %     runInfos.allObj(r) = obj;
        
        
        mInd=mInd+1;
        allRes{mInd}.name = 'LSTM';
        allRes{mInd}.acc(r) = matchAsg(myResMat, asgT);
        allRes{mInd}.obj(r) = resVec(:)' * newK * resVec(:);
        allRes{mInd}.time(r) = resRaw(1,3);
        allRes{mInd}.resMat(:,:,r) = myResMat;
        
        % resolve with hungarian
        resObj = resRaw(:,2);
        resMat = reshape(resObj,N,N)';
        thun = tic;
        [matchHun, costHun] = hungarian(-resMat);
        thun=toc(thun);
        hunVec = reshape(matchHun',N*N,1);
        %     runInfos.rnnHunTime(r) = runInfos.rnnTime(r) + thun;
        %     runInfos.rnnHunObj(r) = hunVec' * newK * hunVec;
        %     runInfos.rnnHunAcc(r) = matchAsg(matchHun, asgT);
        mInd=mInd+1;
        allRes{mInd}.name = 'LSTM-HA';
        allRes{mInd}.acc(r) = matchAsg(matchHun, asgT);
        allRes{mInd}.obj(r) = hunVec' * newK * hunVec;
        allRes{mInd}.time(r) = resRaw(1,3)+thun;
        allRes{mInd}.resMat(:,:,r) = matchHun;
        
        %     pause
    catch err
        fprintf('WARNING. LSTM IGNORED. %s\n',err.message);
    end
    
    %     genFig(2, r, allFs, allGphsSubSel, randSmpl, asgT, allRes);
    %     genFig(6, r, allFs, allGphsSubSel, randSmpl, asgT, allRes);
    %     genFig(7, r, allFs, allGphsSubSel, randSmpl, asgT, allRes);
    
end

% fprintf('Average RNN Accuracy: %.2f\n',mean(runInfos.allAcc))
% fprintf('Average Gur Accuracy: %.2f\n',mean(runInfos.gurAcc))

%% Stats
mbstMethod = sprintf('%d-bstMar',mBst);
mbstHAMethod = sprintf('%d-bstMarH',mBst);
moptMethod = sprintf('%d-opt',mBst);
fprintf('\n%15s|%8s|%8s|%8s|%8s\n','Method','acc','obj','time','optim.');
fprintf('-----------------------------------------------\n');
% fprintf('%10s|%8.2f|%8.2f|%8.3f\n','IPFP',mean(runInfos.IPFPAcc),mean(runInfos.IPFPObj),mean(runInfos.IPFPTime));
% fprintf('%10s|%8.2f|%8.2f|%8.3f\n','Gurobi',mean(runInfos.gurAcc),mean(runInfos.gurObj),mean(runInfos.gurTime));
% fprintf('%10s|%8.2f|%8.2f|%8.3f\n','LSTM',mean(runInfos.allAcc),mean(runInfos.allObj),mean(runInfos.rnnTime));
% fprintf('%10s|%8.2f|%8.2f|%8.3f\n','LSTM-HUN',mean(runInfos.rnnHunAcc),mean(runInfos.rnnHunObj),mean(runInfos.rnnHunTime));
% fprintf('%10s|%8.2f|%8.2f|%8.3f\n',mbstMethod,mean(runInfos.mbstAcc),mean(runInfos.mbstObj),mean(runInfos.mbstTime));
% fprintf('%10s|%8.2f|%8.2f|%8.3f\n',mbstHAMethod,mean(runInfos.mbstHAAcc),mean(runInfos.mbstHAObj),mean(runInfos.mbstHATime));
% fprintf('%10s|%8.2f|%8.2f|%8.3f\n',moptMethod,mean(runInfos.moptAcc),mean(runInfos.moptObj),mean(runInfos.moptTime));

for mInd=1:length(allRes)
    if strcmp(allRes{mInd}.name,'Branch-and-cut')
        fprintf('%15s|%8.2f|%8.2f|%8.3f|%8.1f %%\n',allRes{mInd}.name, ...
            mean(allRes{mInd}.acc),mean(allRes{mInd}.obj),mean(allRes{mInd}.time),sum(allRes{mInd}.optimal)/length(nRuns)*100);
    else
        fprintf('%15s|%8.2f|%8.2f|%8.3f\n',allRes{mInd}.name,mean(allRes{mInd}.acc),mean(allRes{mInd}.obj),mean(allRes{mInd}.time));
    end
end

%% export to latex
doExport=false;
% doExport=true;
if doExport
    fil = fopen(sprintf('%s/numbers/matching-N%d.tex',getPaperDir,N),'w');
    for mInd=1:length(allRes)
        if strcmp(allRes{mInd}.name,'LSTM'), fprintf(fil, '\\midrule\n'); end
        if isfield(allRes{mInd},'cite')
            allRes{mInd}.name = sprintf('%s \\cite{%s}',allRes{mInd}.name,allRes{mInd}.cite);
        end
        fprintf(fil,'%25s & %8.2f & %8.2f & %8.3f\\\\\n',allRes{mInd}.name,mean(allRes{mInd}.acc),mean(allRes{mInd}.obj),mean(allRes{mInd}.time));
    end
    fclose(fil);
end
