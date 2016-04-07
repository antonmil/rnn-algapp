%function [region_mus, intersect_mus, val, all_mus, A, b, obj, prepTime] = solve_glp_relax(glp_state, A, b, numIneqsAtEnd, obj)
% G is a graph over hidden variables. We want to solve the MAP
% relaxation and obtain the marginals
function [region_mus, intersect_mus, val, all_mus, A, b, obj, prepTime] = solve_glp_relax(glp_state, A, b, numIneqsAtEnd, obj)

local = glp_state.orig_local;
N = length(local);
nVals = zeros(N,1);
for i=1:N
  nVals(i) = length(local{i});
end

% host = getenv('HOST');
% if strcmp(host,'cmos-12')
  solver = 'cplex';
% else
%   solver = 'glpk';
% end

ind = 1;

% Define locations for clique marginals:
numRegions = length(glp_state.regions);
region_start = zeros(numRegions,1);
region_end = zeros(numRegions,1);
for i=1:numRegions
  curr_size = prod(nVals(glp_state.regions{i}));
  region_start(i) = ind;
  region_end(i) = ind+curr_size-1;
  ind = ind + curr_size;
end

% Define locations for intersection marginals:
nIntersects = length(glp_state.intersects);
intersect_start = zeros(nIntersects,1);
intersect_end = zeros(nIntersects,1);
for i=1:nIntersects
  curr_size = prod(nVals(glp_state.intersects{i}));
  intersect_start(i) = ind;
  intersect_end(i) = ind+curr_size-1;
  ind = ind + curr_size;
end
nVars = ind-1;

% Generate LP matrices for the problem:
prepTime = 0;
if (nargin < 3 || isempty(A) || isempty(b))
  fprintf('Starting to prepare A\n'); tic
  prepStartTime = cputime;
  
  nConst = nIntersects;
  nnz = intersect_end(end)-intersect_start(1)+1;
  
  for ri=1:numRegions
    curr_region_subsets = glp_state.region_subsets{ri};
    for si=1:length(curr_region_subsets)
      interInd = curr_region_subsets(si);
      curr_subset_vars = glp_state.intersects{interInd};
      nConst = nConst + prod(nVals(curr_subset_vars));
      nnz = nnz + prod(nVals(glp_state.regions{ri})) + prod(nVals(curr_subset_vars));
    end
  end
  
  %A = sparse(nConst,nVars);
  A_inds = zeros(nnz,3);
  b = zeros(nConst,1);

  ind = 1;
  nnzind = 1;
  
  for ri=1:numRegions
    curr_region_subsets = glp_state.region_subsets{ri};
    vars_in_region = glp_state.regions{ri};
    nvals_in_region = nVals(vars_in_region);
    for si=1:length(curr_region_subsets)
      interInd = curr_region_subsets(si);
      mask = get_marginal_mask(nvals_in_region, glp_state.inds_in_region{ri,si});
      vals_in_inter = size(mask,1);
%      A(ind:ind+vals_in_inter-1,region_start(ri):region_end(ri)) = mask;
      [ii,jj,vv] = find(mask);
      A_inds(nnzind:nnzind+length(ii)-1,:) = [ind+ii(:)-1 region_start(ri)+jj(:)-1 vv(:)];
      nnzind = nnzind+length(ii);
%      A(ind:ind+vals_in_inter-1,intersect_start(interInd):intersect_end(interInd)) = -eye(vals_in_inter);      
      [ii,jj,vv] = find(-eye(vals_in_inter));        
      A_inds(nnzind:nnzind+length(ii)-1,:) = [ind+ii(:)-1 intersect_start(interInd)+jj(:)-1 vv(:)];
      nnzind = nnzind+length(ii);      
      
      b(ind:ind+vals_in_inter-1) = 0;
      ind = ind+vals_in_inter;
    end
  end
  
  
  
  % Add normalization on intersects
  for si=1:nIntersects
%    A(ind,intersect_start(si):intersect_end(si)) = 1;
    b(ind) = 1;
    len = intersect_end(si)-intersect_start(si)+1;
    A_inds(nnzind:nnzind+len-1,:) = [ones(len,1)*ind (intersect_start(si):intersect_end(si))' ones(len,1)];
    nnzind = nnzind+len;
    ind = ind + 1;
  end
  %A = sparse(A);
  A = spconvert(A_inds);
  
  prepEndTime = cputime;
  prepTime = prepEndTime - prepStartTime;
  fprintf('Finished preparing A. Took=%g\n',toc);
  if (size(A,1) ~= nConst || size(b,1) ~= nConst)
    fprintf('%i = size(A,1) ~= nConst = %i || %i = size(b,1) ~= nConst = %i\n', size(A,1), nConst, size(b,1), nConst);
    error('logical error in A, b DIMS!');
  end
end

if (nargin < 4)
  % Note that if nargin < 3, then all of A and b were created in code above.
  % In any case where numIneqsAtEnd is not given, force it to be 0.
  numIneqsAtEnd = 0;
end

if (nargin < 5 || isempty(obj))
  obj = zeros(1,nVars);
  
  for ri=1:numRegions
    if ~isempty(glp_state.lambda{ri})
      obj(region_start(ri):region_end(ri)) = glp_state.lambda{ri};
    end
  end
  
  for si=1:nIntersects
    if ~isempty(glp_state.intersect_lambda{si})
      obj(intersect_start(si):intersect_end(si)) = glp_state.intersect_lambda{si};
    end
  end
end


% Prepare for glpk:
if strcmp(solver,'glpk')
  glpk_param.msglev = 0; % No output
  glpk_param.scale = 0;
  
  var_type = char(ones(length(obj),1)*'C');
  
  lb = zeros(length(obj),1);
  ub = inf(length(obj),1);
  b_save = 0;
  
  % Mark all of A as equalities [Ax = b]:
  ineq_type = char(ones(length(b),1)*'S');
  
  % Mark the relevant inequalities as being Ax <= b:
  ineq_type(end-numIneqsAtEnd+1:end) = char('U');

  LP_SOLVERS = [1 2]; % simplex, interior point
  SOL_IS_OPTIMAL_CODE = 5;
  
  for i=1:length(LP_SOLVERS)
    lpsolver = LP_SOLVERS(i);
    
    %fprintf('Entering GLPKMEX\n'); tic
    [all_mus, val, glpk_status] = glpkmex(1,-obj',A,b',ineq_type',lb,ub,var_type',glpk_param,lpsolver,b_save);
    %fprintf('Done GLPKMEX. Took=%g. Val=%g\n',toc,val); tic    
    if (glpk_status == SOL_IS_OPTIMAL_CODE) % solved the LP to its optimum
      break;
    else
      disp(['glpk_status = ', num2str(glpk_status), ' ~= ', num2str(SOL_IS_OPTIMAL_CODE), ': unable to find optimal solution!']);
      disp(['LP solver # ', num2str(lpsolver), ' failed!']);
      if (i == length(LP_SOLVERS))
	error('ALL LP solvers failed -- terminating!');
      else
	disp(['trying LP solver # ', num2str(LP_SOLVERS(i+1))]);
      end
    end
  end
elseif strcmp(solver,'cplex')
  b_u = b';
  b_l = b';
  b_l(end-numIneqsAtEnd+1:end) = -1e12;
  lb = zeros(length(obj),1);
  ub = ones(length(obj),1)*1e12;
  if(numIneqsAtEnd > 0)
  Aieq = A(end - numIneqsAtEnd + 1:end, :);
  bieq = b_u((end - numIneqsAtEnd + 1):end)';
  else
      Aieq = [];
      bieq = [];
  end
  useA = A(1:(end - numIneqsAtEnd), :);
  useb = b_u(1:(end - numIneqsAtEnd));
  %fprintf('Entering CPLEX\n'); tic',
%   [all_mus,slack,v,rc,val,ninf,sinf,Inform] = cplex(-obj',A,lb,ub,b_l,b_u);
tic
  [all_mus, val, exitflag] = cplexlp(-obj', Aieq, bieq, useA, useb', lb, ub);
toc
  %val
  %fprintf('Done CPLEX. Took=%g. Val=%g\n',toc,val); tic      
  if exitflag~=1
    error(['cplex status = ', num2str(Inform), ': unable to find optimal solution!']);
  end
else
  error('Unknown solver');
end

  
  
region_mus = cell(numRegions,1);
for ri = 1:numRegions
  region_mus{ri} = all_mus(region_start(ri):region_end(ri));
end

intersect_mus = cell(nIntersects,1);
for si=1:nIntersects
  intersect_mus{si} = all_mus(intersect_start(si):intersect_end(si));
end

% Since glpkmex() solves the minimization of -(objective function):
val = -val;
