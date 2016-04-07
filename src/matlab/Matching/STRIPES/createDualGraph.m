%function [dual_pair_inds] = createDualGraph(pair_inds)
function [dual_pair_inds] = createDualGraph(pair_inds)

numVars = size(pair_inds,1);

[row_inds, col_inds] = find(triu(pair_inds));
numEdges = length(row_inds);

% dual variables are [in this order]:
% 1. all variables (separators)
% 2. all edges (cliques)
numDualVars = numVars + numEdges;
dual_adj = sparse(numDualVars,numDualVars);
edgeOffset = numVars;

for ei = 1:numEdges
  var1 = row_inds(ei);
  var2 = col_inds(ei);
  edgeInd = pair_inds(var1,var2);
  
  dual_adj(var1, edgeOffset + edgeInd) = 1;
  dual_adj(var2, edgeOffset + edgeInd) = 1;
end

[dual_pair_inds, dual_adj] = convertAdjToPairInds(dual_adj);
