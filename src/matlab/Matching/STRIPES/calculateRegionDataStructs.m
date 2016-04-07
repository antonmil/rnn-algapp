%function [region_start, intersect_start, pair_start, single_start, numMuVars, nVals] = calculateRegionDataStructs(glp_state, pair_inds)
function [region_start, intersect_start, pair_start, single_start, numMuVars, nVals] = calculateRegionDataStructs(glp_state, pair_inds)

local = glp_state.orig_local;
N = length(local);
nVals = zeros(N,1);
for i=1:N
  nVals(i) = length(local{i});
end

ind = 1;

% Define locations for clique marginals:
numRegions = length(glp_state.regions);
region_start = zeros(numRegions,1);

numEdges = length(find(triu(pair_inds)));
pair_start = zeros(numEdges,1);

for i=1:numRegions
  reg = glp_state.regions{i};
  curr_size = prod(nVals(reg));
  region_start(i) = ind;
  
  if (length(reg) == 2)
    var1 = reg(1);
    var2 = reg(2);
    edgeInd_in_graph = pair_inds(var1,var2);
    if (edgeInd_in_graph) %otherwise, this is not an original edge
      pair_start(edgeInd_in_graph) = ind;
    end
  end
  
  ind = ind + curr_size;
end

% Define locations for intersection marginals
% (and note singles and pairs as well):
nIntersects = length(glp_state.intersects);
intersect_start = zeros(nIntersects,1);

single_start = zeros(N,1);

for i=1:nIntersects
  inter = glp_state.intersects{i};
  curr_size = prod(nVals(inter));
  intersect_start(i) = ind;
  
  if (length(inter) == 1)
    var = inter(1);
    single_start(var) = ind;
  end
  
  ind = ind + curr_size;
end

if (~isempty(find(single_start == 0, 1)))
  error('Missing single variables from intersects!');
end
if (~isempty(find(pair_start == 0, 1)))
  error('Missing pairwise edges from regions!');
end

numMuVars = ind - 1;
