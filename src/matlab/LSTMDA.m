function [da, runtime, daHun] = LSTMDA(probs, marginals)

% probs = rand(5);

N=size(probs,1);
rnnSize = 128;
numLayers = 1;
solIndex = 2; % 1=integer, 2=distribution
infIndex = 2; % 1=map, 2=marginal
model_sign = sprintf('mt1_r%d_l%d_n%d_m%d_o1_s%d_i%d_valen',rnnSize, numLayers, N,N, solIndex, infIndex);
model_name = 'trainHun';
% model_name = '0509LBc-2'; % trained on JPDA
% model_name = '0511LBc-1'; % trained on JPDA_10
model_name = '0513LAc-1'; % trained on JPDA_10

allQ=reshape(probs', 1, N*N);
allSol=reshape(eye(N), 1, N*N);
if nargin==2, allSol=reshape(marginals',1,  N*N); end
allMarginals=allSol;

testfilebase='da';
testfile = sprintf('%sdata/%s_%d.mat',getRootDir,testfilebase,N);
save(testfile,'all*');

try
    cmd = sprintf('cd ../..; pwd; th %s.lua -model_name %s -model_sign %s -suppress_x 1 -test_file %s','test', ...
        model_name , model_sign, testfilebase);
    [a,b] = system(cmd);
%     b
    if a~=0, fprintf('Error running RNN!\n'); end
catch err
    fprintf('WARNING. LSTM IGNORED. %s\n',err.message);
end

resRaw = dlmread(sprintf('../../../out/%s_%s.txt',model_name, model_sign));
runtime = resRaw(1,3);
%     resRaw(:,1) = reshape(reshape(resRaw(:,1),N,N)',N*N,1);
da = resRaw(:,2);
da = reshape(da, N, N)';

[daHun, ~] = hungarian(-log(da));
