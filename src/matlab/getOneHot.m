function [mat, ass] = getOneHot(vec, N, M)
% converts a binary vector to an assignment matrix
% and extracts assignments as an N-sized integer vector

if nargin<2,    N=sqrt(length(vec)); M=N; end

vec = binarize(vec);

mat = reshape(vec, N, M)';
ass = getIntSol(mat);

end