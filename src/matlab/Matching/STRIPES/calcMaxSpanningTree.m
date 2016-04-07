% Finds the maximal spanning tree over the edge weights for the standard spanning tree inequalities:
function [tree_edges, tree_pair_inds, tree_weight, tree_num_conn_comp] = ...
    calcMaxSpanningTree(elim_assign, pair_inds, all_mus, pair_mus, single_mus)

N = length(pair_inds);
pair_weights = sparse(size(pair_inds,1), size(pair_inds,2));
const_singles_sum = 0;

if (~isempty(all_mus))
    [row_inds, col_inds] = find(triu(pair_inds));
    nEdges = length(row_inds);
    for ei = 1:nEdges
        var1 = row_inds(ei);
        var2 = col_inds(ei);
        [val1, val2] = deal(elim_assign(var1), elim_assign(var2));
        
        edgeWeight = pair_mus{var1,var2}(val1, val2) - (single_mus{var1}(val1) + single_mus{var2}(val2));
        pair_weights(var1,var2) = edgeWeight;
        pair_weights(var2,var1) = edgeWeight;
    end
    
    for i = 1:N
        const_singles_sum = const_singles_sum + single_mus{i}(elim_assign(i));
    end
end

[tree_adj, tree_edges, tree_pair_inds, tree_weight, tree_num_conn_comp] = ...
    UndirectedMaximumSpanningTree(pair_inds, pair_weights);
tree_weight = tree_weight + const_singles_sum;