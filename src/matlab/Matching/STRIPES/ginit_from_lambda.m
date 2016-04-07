%function gmplp_state = ginit_from_lambda(adj, lambda, local, ADD_ALL_EDGES_TO_INTERSECTS)
function gmplp_state = ginit_from_lambda(adj, lambda, local, ADD_ALL_EDGES_TO_INTERSECTS)

if (nargin < 4)
  ADD_ALL_EDGES_TO_INTERSECTS = false;
end

[row_inds, col_inds] = find(triu(adj));

% For the regions, only the pairwise graph edges:
regions = {};
my_region_lambda = {};

numPairs = length(row_inds);
if (numPairs > 0)
  regions = mat2cell([row_inds(:), col_inds(:)], ones(1,numPairs), 2);
  
  my_region_lambda = cell(numPairs,1);
  for i=1:numPairs
    tmp = lambda{row_inds(i), col_inds(i)}';
    my_region_lambda{i} = tmp(:)';
  end
end

% For the intersects, only the singleton nodes:
numNodes = size(adj,1);
intersects = mat2cell([1:numNodes]', ones(1,numNodes), 1);

my_inter_lambda = cell(numNodes,1);
for i=1:numNodes
  my_inter_lambda{i} = local{i}';
end

if (ADD_ALL_EDGES_TO_INTERSECTS && numPairs > 0)
  intersects = {intersects{:}, regions{:}}; % intersects is now: nodes, edges
  my_inter_lambda{numNodes + numPairs} = []; % extend my_inter_lambda to add empty lambdas for edge intersects
end

gmplp_state = gmplp_init(regions,intersects,my_region_lambda,my_inter_lambda,local,0,0,lambda);
