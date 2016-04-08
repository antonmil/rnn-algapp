function torchSparseTensor = sparseTensor(Q)
%% convert matrix Q to a 2D sparse tensor for torch


[u,v]=find(Q'); % transposed for torch vs. matlab indexing
spInd = sub2ind(size(Q),u,v);

% final 2xN tensor
% [ind1, ind2, ind3, ...
%  val1, val2, val3, ...]
torchSparseTensor = [spInd Q(~~Q)]';

end