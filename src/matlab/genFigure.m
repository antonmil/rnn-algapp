% clear all
% close all
% clc

addpath('Matching/showres/');
addpath(genpath('.'));



% nOut = 1;
% kFs=25;% car 20 motor 20

% data_name='CarResult';
% d_name0=['./Matching/Results/',data_name,'_MbstFill_Smple',num2str(kFs),'_Outlier',num2str(nOut),'.mat'];
% d_name0='./Matching/Results/CarResult_Mbst_Smple25_Outlier1.mat';
% d_name0='./Matching/Results/CarResult_Mbst_Smple25_Outlier1.mat';
% load(d_name0)
% load(data_name,'solutionsBPMbst')
% asgmbstBP.X=solutionsBPMbst{nOut+1,kFs}(:,:,50);
% asgmbstBP.X=asgIpfpSMbst{nOut+1,kFs}(:,:,50);
% asgm.X = asgIpfpSMbst.X(:,:,1);
% asgm.X = asgBPMbst.Xmbst;

%%
Fs = allFs{randSmpl};
gphs = allGphsSubSel{randSmpl};
hfigg=figure;
rows = 1; cols = 1;
Ax = iniAx(1, rows, cols, [400 * rows, 900 * cols], 'hGap', .1, 'wGap', .1);
parCor = st('cor', 'ln', 'mkSiz', 7, 'cls', {'y', 'b', 'g'});
shAsgImg(Fs, gphs, asgm, asgT, parCor , 'ax', Ax{1}, 'ord', 'n');
% title('result of mbst-BP');

% print(hfigg,'-dpsc',[data_name,'_mbstBP_matchingresults.eps'])