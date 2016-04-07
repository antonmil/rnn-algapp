%function [pair_inds, adj] = convertAdjToPairInds(adj)
% pair_inds has the same "structure" as adj, but it will contain the ordered
% indices of the edges [non-zero entries in adj], sorted by:
% [row_inds, col_inds] = find(triu(adj))
function [pair_inds, adj] = convertAdjToPairInds(adj)


% Make the adjacency matrix symmetric:
adj = adj + adj';

% Ensure that adj is binary [0 or 1]:
adj(find(adj)) = 1;

% Create pair_inds:
pair_inds = sparse(size(adj,1),size(adj,2));
[row_inds, col_inds] = find(triu(adj));
cur_ind = 1;
for ei = 1:length(row_inds)
  var1 = row_inds(ei);
  var2 = col_inds(ei);
  
  pair_inds(var1,var2) = cur_ind;
  cur_ind = cur_ind + 1;
end

% Make the pair_inds matrix symmetric:
pair_inds = pair_inds + pair_inds';
