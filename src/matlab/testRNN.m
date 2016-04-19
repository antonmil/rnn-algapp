%%
addpath(genpath('.'))
nRuns = 10;


N=10;
rnnSize = 512;
numLayers = 4;
[gurModel, gurParams] = getGurobiModel(N);
model_sign = sprintf('mt1_r%d_l%d_n%d_m%d_o2_s1_i1',rnnSize, numLayers, N,N);
model_name = 'trainHun';

runInfos.allAcc=zeros(1,nRuns);
runInfos.gurAcc=zeros(1,nRuns);
runInfos.allObj=zeros(1,nRuns);
runInfos.gurObj=zeros(1,nRuns);
asgT.X=eye(N);

for r = 1:nRuns
    RM = Pair_M{1,randi(30)};
    [newK,gurResult] = selectSubset(RM, N, true, gurModel, gurParams);
    [gurMat, gurAss] = getOneHot(gurResult.x);
    if length(unique(gurAss)) ~= length(gurAss)
        fprintf('Gurobi solution not one-to-one!\n')
    end
    
    cmd = sprintf('cd ..; pwd; th %s.lua -model_name %s -model_sign %s -suppress_x 1','test', model_name , model_sign);
    [a,b] = system(cmd);
    if a~=0
        fprintf('Error running RNN!\n'); b
        break;
    end

    resRaw = dlmread(sprintf('../../out/%s_%s.txt',model_name, model_sign));
    resVec = resRaw(:,1);
    [myResMat, myAss] = getOneHot(resVec);
    if length(unique(myAss)) ~= length(myAss)
        fprintf('RNN solution not one-to-one!\n')
    end
    
    obj = resVec(:)' * newK * resVec(:);
    acc = matchAsg(myResMat, asgT);
    fprintf('RNN Accuracy: %.2f\n',acc)        
    runInfos.allAcc(r)=acc;
    runInfos.allObj(r) = obj;
    
    acc = matchAsg(gurMat, asgT);
    obj = gurResult.x(:)' * newK * gurResult.x(:);
    fprintf('Gur Accuracy: %.2f\n\n',acc)
    
    runInfos.gurAcc(r)=acc;
    runInfos.gurObj(r) = obj;
end

fprintf('Average RNN Accuracy: %.2f\n',mean(runInfos.allAcc))
fprintf('Average Gur Accuracy: %.2f\n',mean(runInfos.gurAcc))