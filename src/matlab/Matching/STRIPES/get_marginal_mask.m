function mask = get_marginal_mask(nVals,marg_vars)

nVars = length(nVals);
% Return mask that marginalizes on the first variable

nAllVals = prod(nVals); %(nPerVal)^(nVars);
nSubsetVars = length(marg_vars);
nSubsetVals = prod(nVals(marg_vars)); %^nSubsetVars;
nonmarg_vars = setdiff(1:nVars,marg_vars);
nOtherVals = nAllVals/nSubsetVals; %(nPerVal)^(nVars-nSubsetVars);
nOtherVars = nVars-nSubsetVars;

mask = sparse(nSubsetVals,nAllVals);

for vi=0:nSubsetVals-1
  % Fill in the values whose marginal we want
%  full_assign(marg_vars) = my_dec2base(vi,nPerVal,nSubsetVars)-1;

  full_assign(marg_vars) = my_dec2base_multi(vi,nVals(marg_vars))-1;  
  % Go over the remaining variables and sum their values up
  nrm = 0;
  for vim=0:nOtherVals-1
%    full_assign(nonmarg_vars) = my_dec2base(vim,nPerVal,nOtherVars)-1;
    full_assign(nonmarg_vars) = my_dec2base_multi(vim,nVals(nonmarg_vars))-1;
% Find the index of this in the full distribution
    i = my_base2dec_multi(full_assign,nVals)+1;
    mask(vi+1,i) = 1;
  end
end
