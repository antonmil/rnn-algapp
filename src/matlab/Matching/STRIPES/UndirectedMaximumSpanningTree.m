%function [tree_adj, tree_edges, tree_pair_inds, tree_weight, tree_num_conn_comp] = UndirectedMaximumSpanningTree(pair_inds, pair_weights)
% The function takes pair_inds, pair_weights as input and returns the maximum spanning tree (tree_adj).
% Uses Kruskal's Algorithm
% Extract the edge weights from the cost matrix
% Sort the edges in a non decreasing order of weights 
function [tree_adj, tree_edges, tree_pair_inds, tree_weight, tree_num_conn_comp] = UndirectedMaximumSpanningTree(pair_inds, pair_weights)

n = size (pair_weights,1); %Number of vertices
EdgeWeights = [];         %Edges and corresponding weights
EdgeWeightsCounter = 0;
[edge_row_inds, edge_col_inds] = find(triu(pair_inds));

for ei = 1:length(edge_row_inds)
  [i,j] = deal(edge_row_inds(ei), edge_col_inds(ei));
  EdgeWeightsCounter = EdgeWeightsCounter + 1;
  EdgeWeights(EdgeWeightsCounter,1) = pair_weights(i,j);
  EdgeWeights(EdgeWeightsCounter,2) = i;
  EdgeWeights(EdgeWeightsCounter,3) = j;
end

SortedEdgeWeights = [];
SortedEdgeWeights = sortrows(EdgeWeights);
% First column of SortedEdgeWeights are the weights
% Second and third column are the vertices that the edges connect
m = size(SortedEdgeWeights,1); % number of edges 

% We use the Disjoint sets data structures to detect cycle while adding new
% edges. Union by Rank with path compression is implemented here.

% Assign parent pointers to each vertex. Initially each vertex points to 
% itself. Now we have a conceptual forest of n trees representing n disjoint 
% sets 
global ParentPointer ;
ParentPointer = 0;
ParentPointer(1:n) = 1:n;

% Assign a rank to each vertex (root of each tree). Initially all vertices 
% have the rank zero.
TreeRank = 0;
TreeRank(1:n) = 0;

% Visit each edge in the sorted edges array
% If the two end vertices of the edge are in different sets (no cycle), add
% the edge to the set of edges in minimum spanning tree
MSTreeEdges = [];
MSTreeEdgesCounter = 0; i = m;
while ((MSTreeEdgesCounter < (n-1)) && (i>=1))
%Find the roots of the trees that the selected edge's two vertices
%belong to. Also perform path compression.
    root1=0; root2=0; temproot=0;
    temproot = SortedEdgeWeights(i,2);
    root1 = FIND_PathCompression(temproot);
  
    temproot = SortedEdgeWeights(i,3);
    root2 = FIND_PathCompression(temproot);
    
    if (root1 ~= root2)
        MSTreeEdgesCounter = MSTreeEdgesCounter + 1;
        MSTreeEdges(MSTreeEdgesCounter,1:3) = SortedEdgeWeights(i,:);
        if (TreeRank(root1)>TreeRank(root2))
            ParentPointer(root2)=root1;
        else
            if (TreeRank(root1)==TreeRank(root2))
               TreeRank(root2)=TreeRank(root2) + 1;
            end
            ParentPointer(root1)=root2;
        end
    end
    i = i - 1;
end


%tree_adj = 0;
%tree_adj(1:n,1:n)=0;
tree_adj = sparse(n,n);

numEdgesInForestOfTrees = size(MSTreeEdges,1);
for MSTreeEdgesCounter = 1:numEdgesInForestOfTrees
  node1 = MSTreeEdges(MSTreeEdgesCounter,2);
  node2 = MSTreeEdges(MSTreeEdgesCounter,3);
  
  tree_adj(node1, node2) = 1;
  tree_adj(node2, node1) = 1;
end

tree_edges = find(triu(tree_adj))';
tree_pair_inds = tree_adj .* pair_inds;
tree_weight = sum(sum(triu(tree_adj) .* pair_weights));

% For a forest of trees, the number of edges plus the number of connected components 
% is equal to the number of vertices: |E| + |CC| = |V|.
tree_num_conn_comp = n - numEdgesInForestOfTrees;
