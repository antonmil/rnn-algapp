function [Xm,objm,tme] = gmPosD_m(K, Ct, X0, par)
% Computer a discrete assingment matrix by rounding from a continuous one.
%
% Input
%   K       -  affinity matrix, [] | nn x nn (sparse)
%   Ct      -  constraints, n1 x n2
%   X0      -  continuous correspondence, n1 x n2
%   par     -  parameter
%     alg   -  method, 'gre' | 'hun' | 'ipfp'
%                'gre'  : greedy algorithm
%                'hun'  : hungraian algorithm
%                'ipfp' : integer fixed point algorithm
%
% Output
%   X       -  discrete correspondence, n1 x n2
%


% function parameter
alg = par.alg;

if strcmp(alg, 'ipfp_mbst')
    [Xm,objm,tme] = gmPosDIpfp_mbst(K, Ct, X0, par);

else
    error('unknown method: %s', alg);
end