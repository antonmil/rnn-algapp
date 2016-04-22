%%
addpath(genpath('.'))
nRuns = 10;

rng(1);
if ~exist('Pair_M','var')
    Pair_M=doMatching();
end

N=7;
rnnSize = 64;
numLayers = 2;
solIndex = 2; % 1=integer, 2=distribution
infIndex = 2; % 1=map, 2=marginal
[gurModel, gurParams] = getGurobiModel(N);
model_sign = sprintf('mt1_r%d_l%d_n%d_m%d_o2_s%d_i%d',rnnSize, numLayers, N,N, solIndex, infIndex);
model_name = 'trainHun';
mBst = 1;

runInfos.allAcc=zeros(1,nRuns);
runInfos.gurAcc=zeros(1,nRuns);
runInfos.mbstAcc=zeros(1,nRuns);

runInfos.allObj=zeros(1,nRuns);
runInfos.gurObj=zeros(1,nRuns);
runInfos.mbstObj=zeros(1,nRuns);

runInfos.rnnTime=zeros(1,nRuns);
runInfos.gurTime=zeros(1,nRuns);
runInfos.mbstTime=zeros(1,nRuns);

asgT.X=eye(N);

for r = 1:nRuns
    RM = Pair_M{1,randi(length(Pair_M))};
    [newK,gurResult] = selectSubset(RM, N, true, gurModel, gurParams);
    runInfos.gurTime(r) = gurResult.runtime;
    [gurMat, gurAss] = getOneHot(gurResult.x);
    if length(unique(gurAss)) ~= length(gurAss)
        fprintf('Gurobi solution not one-to-one!\n')
    end
    
    % marg
    asgIpfpSMbst = mBestIPFP(newK,mBst);
    mbstVec = reshape(asgIpfpSMbst.Xmbst',N*N,1);
    runInfos.mbstTime = sum(asgIpfpSMbst.time);
    runInfos.mbstObj = mbstVec' * newK * mbstVec;
    runInfos.mbstAcc = matchAsg(asgIpfpSMbst.Xmbst, asgT);
    
    cmd = sprintf('cd ..; pwd; th %s.lua -model_name %s -model_sign %s -suppress_x 1','test', model_name , model_sign);
    [a,b] = system(cmd);    
    if a~=0
        fprintf('Error running RNN!\n'); b
        break;
    end    

    resRaw = dlmread(sprintf('../../out/%s_%s.txt',model_name, model_sign));
    runInfos.rnnTime(r) = resRaw(1,3);
%     resRaw(:,1) = reshape(reshape(resRaw(:,1),N,N)',N*N,1);
    resVec = resRaw(:,1);
    [myResMat, myAss] = getOneHot(resVec);
    if length(unique(myAss)) ~= length(myAss)
        fprintf('RNN solution not one-to-one!\n')
    end
    
    obj = resVec(:)' * newK * resVec(:);
    acc = matchAsg(myResMat, asgT);
%     fprintf('RNN Accuracy: %.2f\n',acc)        
    runInfos.allAcc(r)=acc;
    runInfos.allObj(r) = obj;
    
    acc = matchAsg(gurMat, asgT);
    obj = gurResult.x(:)' * newK * gurResult.x(:);
%     fprintf('Gur Accuracy: %.2f\n\n',acc)
    
    runInfos.gurAcc(r)=acc;
    runInfos.gurObj(r) = obj;
end

% fprintf('Average RNN Accuracy: %.2f\n',mean(runInfos.allAcc))
% fprintf('Average Gur Accuracy: %.2f\n',mean(runInfos.gurAcc))

%% Stats
mbstMethod = sprintf('%d-bst',mBst);
fprintf('\n%10s|%8s|%8s|%8s\n','Method','acc','obj','time');
fprintf('-------------------------------------\n');
fprintf('%10s|%8.2f|%8.2f|%8.3f\n','Gurobi',mean(runInfos.gurAcc),mean(runInfos.gurObj),mean(runInfos.gurTime));
fprintf('%10s|%8.2f|%8.2f|%8.3f\n','LSTM',mean(runInfos.allAcc),mean(runInfos.allObj),mean(runInfos.rnnTime));
fprintf('%10s|%8.2f|%8.2f|%8.3f\n',mbstMethod,mean(runInfos.mbstAcc),mean(runInfos.mbstObj),mean(runInfos.mbstTime));