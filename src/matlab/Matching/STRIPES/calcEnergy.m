%function [energy] = calcEnergy(assign, adj, lambda, local)
% Calculate energy for assignment:
function [energy] = calcEnergy(assign, adj, lambda, local)

energy = 0;
nNodes = length(adj);

for i=1:nNodes
  energy = energy + local{i}(assign(i));
  neighbs = find(adj(i,:));
  for jInd = 1:length(neighbs)
    j = neighbs(jInd);
    if (j < i)
      continue;
    end
    energy = energy + lambda{i,j}(assign(i), assign(j));
  end
end
