%function [region_sep_pair_inds] = createRegionSeparatorGraph(glp_state)
function [region_sep_pair_inds] = createRegionSeparatorGraph(glp_state)

numSeps = length(glp_state.intersects);
numRegions = length(glp_state.regions);

% variables are [in this order]:
% 1. all separators
% 2. all regions
numVars = numSeps + numRegions;
region_sep_adj = sparse(numVars,numVars);
edgeOffset = numSeps;

for i=1:numRegions
  curr_region_subsets = glp_state.region_subsets{i};
  region_sep_adj(curr_region_subsets, edgeOffset + i) = 1;
end

% region_sep_pair_inds, region_sep_adj are symmetric:
[region_sep_pair_inds, region_sep_adj] = convertAdjToPairInds(region_sep_adj);
