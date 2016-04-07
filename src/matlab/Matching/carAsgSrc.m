function wsSrc = carAsgSrc(pFs, nOus)
% Generate CMU Motion source for assignment problem.
%
% Input
%   tag      -  'house' | 'hotel'
%   pFs      -  frame index, 1 x 2
%   nIns     -  #inliers, 1 x 2, [1~30, 1~30]
%   varargin
%     save option
%
% Output
%   wsSrc
%     prex   -  prex
%     asgT   -  ground truth assignment
%     Pts    -  graph node set, 1 x mG (cell), 2 x ni
%     ords   -  order, 1 x mG (cell), 1 x ni
%
% History
%   create   -  Feng Zhou (zhfe99@gmail.com), 08-11-2011
%   modify   -  Feng Zhou (zhfe99@gmail.com), 03-03-2013

% save option
%prex = cellStr(tag, pFs, nIns);
Fname = sprintf('%s_%d.mat', ...
    './Cars_and_Motorbikes_Dataset_and_Code/Data_for_Cars_and_Motorbikes/Data_Pairs_Cars/pair', pFs);
prex = sprintf('Motor_pair_%d', pFs);
load(Fname);
nIn = nF1;

Inliers1 = features1(1:nIn, 1:2)';
Inliers2 = features2(1:nIn, 1:2)';
% load
[n1,~] = size(features1);
[n2,~] = size(features2);
n1 = n1 - nIn;
n2 = n2 - nIn;
nOus = min(min(n1, n2), nOus);
ord1 = randperm(n1);
ord1 = ord1(1:nOus);
Outliers1 = features1(nIn + ord1, 1:2)';
ord2 = randperm(n2);
ord2 = ord2(1:nOus);
Outliers2 = features2(nIn + ord2, 1:2)';

Pts{1} = [Inliers1, Outliers1];
Pts{2} = [Inliers2, Outliers2];

% marker position
%Pts = CMUM.(tag).XTs(pFs);

% ground-truth assignment
XT = [eye(nIn), zeros(nIn, nOus); zeros(nOus, nIn +  nOus)] ;
asgT.alg = 'truth';
asgT.X = XT;
ords = {[1:nIn, ord1 + nIn], [1:nIn, ord2  + nIn]};
Features={features1, features2};
Fs={I1,I2};
% store
wsSrc.prex = prex;
wsSrc.Pts = Pts;
wsSrc.asgT = asgT;
wsSrc.ords = ords;
wsSrc.tag = 'Motor';
wsSrc.pFs = pFs;
wsSrc.nIns = nIn;
wsSrc.Features = Features;
wsSrc.Fs = Fs;
% save


prOut;
