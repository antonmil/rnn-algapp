%% Methods & Settings
% Script for setting algorithms to run
%
% Yumin Suh, Minsu Cho, and Kyoung Mu Lee, 
% Graph Matching via Sequential Monte Carlo, 
% Proc. European Conference on Computer Vision (ECCV), 2012
% http://cv.snu.ac.kr/research/~SMCM/
% Computer Vision Lab, Seoul National University, Korea

% You can add an algorithm following the script below
%nMethods = 1;
%methods(nMethods).fhandle = @fhandle;                         % Function of the algorithm
%methods(nMethods).variable = {'var1', 'var2', 'var3'};        % Input variables that the algorithm requires
%methods(nMethods).param = {'name1', 'val1', 'name2', 'val2'}; % Default parameter values
%methods(nMethods).strName = 'algorithm name';                 % Algorithm name tag
%methods(nMethods).color = 'color';                            % Color for plots
%methods(nMethods).lineStyle = 'line style';                   % Line style for plots
%methods(nMethods).marker = 'marker';                          % Marker for plots

nMethods = 0;
%% SMCM Suh et al. ECCV 2012
if 1
    nMethods = nMethods + 1;
    methods(nMethods).fhandle = @SMCM;
    methods(nMethods).variable = {'affinityMatrix', 'group1', 'group2'};
    methods(nMethods).param = {'nParticles', 2000, 'tau', 2};
    methods(nMethods).postProcess = 'none';
    methods(nMethods).strName = 'SMCM';
    methods(nMethods).color = 'r';
    methods(nMethods).lineStyle = '-';
    methods(nMethods).marker = 'o';
end
%% RRWM Cho et al. ECCV 2010
if 1
    nMethods = nMethods + 1;
    methods(nMethods).fhandle = @RRWM;
    methods(nMethods).variable = {'affinityMatrix', 'group1', 'group2'};
    methods(nMethods).param = {};
    methods(nMethods).postProcess = 'hungarian';
    methods(nMethods).strName = 'RRWM';
    methods(nMethods).color = 'b';
    methods(nMethods).lineStyle = '--';
    methods(nMethods).marker = 'x';
end
%% Spectral Matching Leordeanu et al. ICCV 2005
if 1
    nMethods = nMethods + 1;
    methods(nMethods).fhandle = @SM;
    methods(nMethods).variable = {'affinityMatrix'};
    methods(nMethods).param = {};
    methods(nMethods).postProcess = 'hungarian';
    methods(nMethods).strName = 'SM';
    methods(nMethods).color = 'k';
    methods(nMethods).lineStyle = '--';
    methods(nMethods).marker = 'x';
end

%% Show the algorithms to run
disp('* Algorithms to run *');
for k = 1:nMethods, disp([methods(k).strName ' : @' func2str(methods(k).fhandle)]); end; disp(' ')
clear k