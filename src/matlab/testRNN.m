%%
nRuns = 1;
allAcc=zeros(1,nRuns);
N=5;
model = getGurobiModel(N);


for r = 1:nRuns
    newK = selectSubset(Pair_M{1,randi(30)}, N, true, model, params);
    model_sign = sprintf('mt1_r128_l1_n%d_m%d_o2_s1_i1',N,N);
    model_name = 'trainHun';
    cmd = sprintf('cd ..; pwd; th %s.lua -model_name %s -model_sign %s -suppress_x 1','test', model_name , model_sign);
    [a,b] = system(cmd);

    res = dlmread(sprintf('../../out/%s_%s.txt',model_name, model_sign));
    myres=reshape(res(:,1),N,N);
    [u,v]=find(myres'); u'
    myres(:)' * newK * myres(:)
    asgT.X=eye(N);
    acc = matchAsg(myres, asgT);
    fprintf('Accuracy: %.2f\n',acc)

    allAcc(r)=acc;
end
fprintf('Average Accuracy: %.2f\n',mean(allAcc))