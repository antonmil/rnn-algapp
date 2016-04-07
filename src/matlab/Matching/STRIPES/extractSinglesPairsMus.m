%function [single_mus, pair_mus] = extractSinglesPairsMus(intersect_mus, region_mus, glp_state, nVals, adj)
% Extract the reshaped single and pair mus from the intersect mus:
function [single_mus, pair_mus] = extractSinglesPairsMus(intersect_mus, region_mus, glp_state, nVals, adj)

N = length(nVals);
single_mus = cell(N,1);
pair_mus = cell(N,N);

intersects = glp_state.intersects;
numIntersects = length(intersects);
for intInd=1:numIntersects
  inter = intersects{intInd};

  % Singleton variables are all in the intersects:
  if (length(inter) == 1)
    var = inter(1);
    single_mus{var} = intersect_mus{intInd};
  end
end

regions = glp_state.regions;
numRegions = length(regions);
for regInd=1:numRegions
  reg = regions{regInd};

  % Pair edges are all in the regions:
  if (length(reg) == 2)
    var1 = reg(1);
    var2 = reg(2);

    % otherwise, this is not an original edge:
    if (adj(var1,var2))
      % Since separator assignments are "flattened" [by get_marginal_mask] 
      % in REVERSE order of the variables within the separator:
      pair_mus_mat = reshape(region_mus{regInd}, nVals(var2), nVals(var1))';
      
%      pairNumVals = nVals([var1 var2]);
%      test_mat = zeros(pairNumVals');
%      numPairAssigns = prod(pairNumVals);
%      for x = 1:numPairAssigns
%	pairAssign = my_dec2base_multi(x-1, pairNumVals);
%	flat_ind = my_base2dec_multi(pairAssign - 1, pairNumVals) + 1;
%	test_mat(pairAssign(1), pairAssign(2)) = region_mus{regInd}(flat_ind);
%      end
%      if (~isempty(find(test_mat ~= pair_mus_mat,1)))
%	error('test_mat ~= pair_mus_mat');
%      end
      
      pair_mus{var1,var2} = pair_mus_mat;
      pair_mus{var2,var1} = pair_mus_mat';
    end
  end
end
