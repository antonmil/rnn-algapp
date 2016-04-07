%function [MPLP_solutions, MPLP_energies, iterRunTimes] = GMPLP_top_M(pair_inds, lambda, local, glp_state, M, NEXT_BEST_METH, SAVE_STATS_NAME, DEBUG)
%
% pair_inds = symmetric graph matrix, where pair_inds(i,j) is the INDEX of
% the edge between i and j (0 if no edge), where convertAdjToPairInds()-ORDERED indices start from 1.
%
% M = number of top solutions desired
%
% NEXT_BEST_METH: 'STRIPES', 'Nilsson'
%
function [MPLP_solutions, MPLP_energies, iterRunTimes] = GMPLP_top_M(pair_inds, lambda, local, glp_state, ...
    M, NEXT_BEST_METH, SAVE_STATS_NAME, DEBUG)

if (nargin < 4 || isempty(glp_state))
    [glp_state] = ginit_from_lambda(pair_inds, lambda, local);
end
if (nargin < 5)
    M = 100;
end
if (nargin < 6)
    NEXT_BEST_METH = 'STRIPES';
end
if (nargin < 7)
    SAVE_STATS_NAME = '';
end
if (nargin < 8)
    DEBUG = false;
end

if (~isempty(SAVE_STATS_NAME))
    SAVE_STATS_NAME = [SAVE_STATS_NAME, '.mat'];
end

requirePairIndsIsValidOrExit(pair_inds);

% N = number of variables in the graph:
N = length(local);

[region_start, intersect_start, pair_start, single_start, numMuVars, nVals] = calculateRegionDataStructs(glp_state, pair_inds);

iterRunTimes = [];

MPLP_solutions = [];
MPLP_energies  = [];

% Helper data structures:
lock_constraints = {}; % lock_constraints{n} contains rows of:
% [1/0 var val], where 1 indicates a positive constraint
% and 0 a negative constraint

next_best_energies = [];
next_best_solutions = [];

next_best_vars_vals = [];

NEG_LOCK = 0;
POS_LOCK = 1;

REQUIRE_FIRST_MAP_INTEGRAL = false;

for m = 1:M
    iterStartTime = cputime;
    if (DEBUG)
        %
        %unix('ps ux --sort -pcpu');
        %
        disp(['m = ', num2str(m)]);
    end
    if (m == 1)
        lock_constraints{1} = zeros(0,3); % ENSURE that it has 3 columns
        
        [region_mus, intersect_mus, enVal_ub, all_mus, A, b, obj, prepTime] = solve_glp_relax(glp_state);
        iterStartTime = iterStartTime + prepTime; % "Push off" start time by the overhead of the LP preparation time
        
        % perform the 1st MAP decoding:
        [x, enVal, fractional] = decodeMus(enVal_ub, all_mus, intersect_mus, region_mus, nVals, 'First MAP', glp_state, pair_inds, lambda, local, DEBUG);
        
        % 1st MAP is fractional:
        if (REQUIRE_FIRST_MAP_INTEGRAL && fractional)
            return;
        end
        
        MPLP_solutions(m,:) = x;
        MPLP_energies(m,:)  = enVal;
        
        stats.num_tree_ineqs = 0;
        stats.fractional = fractional;
        iterStats{m,1} = stats;
    else
        [NB_enVal, t_m] = max(next_best_energies);
        if (isinf(NB_enVal))
            error('MPLP_top_M LOGICAL ERROR: next best energy value == -inf [All solutions have been exhausted]');
        end
        
        MPLP_solutions(m,:) = next_best_solutions(t_m,:);
        MPLP_energies(m,:)  = NB_enVal;
        
        if (m < M)
            [lock_var, lock_val] = deal(next_best_vars_vals(t_m,1), next_best_vars_vals(t_m,2));
            lock_constraints{m} = lock_constraints{t_m};
            
            % Remove negative constraints from m that referred to the now-positively constrained variable
            % [since now they're unnecessary]:
            subspace_m_descripts = lock_constraints{m};
            if (~isempty(subspace_m_descripts))
                allLockedVars = subspace_m_descripts(:,2);
                negLocksOnVar = (find(allLockedVars == lock_var));
                if (DEBUG)
                    if (~isempty(find(subspace_m_descripts(negLocksOnVar,1) == POS_LOCK, 1)) ... % there should not be any POSITIVE locks
                            || ~isempty(find(subspace_m_descripts(negLocksOnVar,3) == lock_val, 1))) % should not be negatively locked against the new postively locked value
                        error('LOGICAL error in STRIPES partitioning scheme!');
                    end
                end
                lock_constraints{m}(negLocksOnVar, :) = [];
            end
            % Add positive constraint to m:
            lock_constraints{m}(end+1,:) = createLockConstraint(POS_LOCK, lock_var, lock_val, single_start, numMuVars);
            
            % Add negative constraint to t_m:
            lock_constraints{t_m}(end+1,:) = createLockConstraint(NEG_LOCK, lock_var, lock_val, single_start, numMuVars);
            
            % Update the next best solution in subspace t_m:
            [next_best_solutions(t_m,:), next_best_energies(t_m), next_best_vars_vals(t_m,:), iterStats{m,1}] = ...
                calcNextBest(MPLP_solutions(t_m,:), lock_constraints{t_m}, ...
                glp_state, pair_inds, lambda, local, A, b, obj, NEXT_BEST_METH, DEBUG);
        end
    end
    if (m < M)
        % Update the next best solution in subspace m:
        [next_best_solutions(m,:), next_best_energies(m), next_best_vars_vals(m,:), iterStats{m,2}] = ...
            calcNextBest(MPLP_solutions(m,:), lock_constraints{m}, ...
            glp_state, pair_inds, lambda, local, A, b, obj, NEXT_BEST_METH, DEBUG);
    end
    
    iterEndTime = cputime;
    iterRunTimes(m,:) = iterEndTime - iterStartTime;
    
    if (~isempty(SAVE_STATS_NAME))
        save(SAVE_STATS_NAME, 'MPLP_solutions', 'MPLP_energies', 'iterRunTimes', 'iterStats');
    end
end



% Calculates the 2nd-best solution within sub-space n:
function [next_best_solutions_n, next_best_energies_n, next_best_vars_vals_n, stats] = ...
    calcNextBest(x_n, space_n_lock_constraints, glp_state, pair_inds, lambda, local, A, b, obj, NEXT_BEST_METH, DEBUG)

% Flag to save memory (but no error checking that trees should be UNIQUE!):
SAVE_MEMORY = false;

stats = [];

NEG_LOCK = 0;
POS_LOCK = 1;

APPROX_ZERO_THRESH = 1e-5;

[region_start, intersect_start, pair_start, single_start, numMuVars, nVals] = calculateRegionDataStructs(glp_state, pair_inds);

N = length(local);
issetpos = cell(N,1);
issetneg = cell(N,1);
for vi=1:N
    nVals(vi) = length(local{vi});
    issetpos{vi} = zeros(nVals(vi),1);
    issetneg{vi} = zeros(nVals(vi),1);
end

if (DEBUG)
    disp('START partition locks:');
end
num_lock_constraints = size(space_n_lock_constraints,1);
for c=1:num_lock_constraints
    constraint = space_n_lock_constraints(c,:);
    [pos_neg, var, val] = deal(constraint(1), constraint(2), constraint(3));
    if (DEBUG)
        isNegChar = '';
        if (pos_neg == NEG_LOCK)
            isNegChar = '!';
        end
        fprintf('x_%i %s= %i\n', var, isNegChar, val);
    end
    if (pos_neg)
        issetpos{var}(val) = 1;
    else
        issetneg{var}(val) = 1;
    end
end
if (DEBUG)
    disp('END partition locks.');
end

numSetVars = 0;
for vi=1:N
    if sum(issetpos{vi}) > 0
        numSetVars = numSetVars+1;
    elseif sum(~issetneg{vi}) == 1
        numSetVars = numSetVars+1;
    end
end
if (numSetVars == N) % LP will be infeasible:
    [next_best_solutions_n, next_best_energies_n, next_best_vars_vals_n] = ...
        deal(-ones(1,N), -inf, [-1 -1]);
    return;
end

[n_lock_pos_negs, n_lock_vars, n_lock_vals] = deal(space_n_lock_constraints(:,1), space_n_lock_constraints(:,2), space_n_lock_constraints(:,3));
[space_n_lock_constraints_eqs, space_n_lock_constraints_eq_consts] = createLockEqualities(n_lock_vars, n_lock_vals, n_lock_pos_negs, single_start, numMuVars);

if (strcmp(NEXT_BEST_METH, 'STRIPES'))
    FALLBACK_ON_NILSSON_WHEN_FRACTIONAL = true;
    RESORT_TO_NILSSON_AFTER_N_TREES = false;
    
    use_A = [A; space_n_lock_constraints_eqs];
    use_b = [b; space_n_lock_constraints_eq_consts];
    
    TRY_DUAL_CLIQUE_TREE_INEQUALITIES = true;
    dual_pair_inds = [];
    
    TRY_REGION_SEP_TREE_INEQUALITIES = true;
    region_sep_pair_inds = [];
    
    treeInequalityMode = 'standard';
    
    % Add spanning tree inequalities to enforce: x != x_n:
    [all_mus, pair_mus, single_mus, region_mus, intersect_mus] = deal([],[],[],[],[]);
    failed = false;
    
    numAddedIneq = 0;
    
    added_trees = [];
    added_dual_trees = [];
    added_region_trees = {};
    
    fractional = true;
    while (isempty(all_mus) || fractional)
        if (RESORT_TO_NILSSON_AFTER_N_TREES && size(added_trees,1) >= N) % revert to Nilsson if already added N trees
            failed = true;
            disp('Added N trees, so trying Nilsson''s method instead...');
            NEXT_BEST_METH = 'Nilsson';
            break;
        end
        
        % ONLY NECESSARY FOR NOW SINCE THESE ARE USED TO CREATE THE "STANDARD"
        % SPANNING TREE INEQUALITIES [WHERE POTENTIAL REGIONS ARE ALL EDGES]:
        if (~isempty(all_mus))
            [single_mus, pair_mus] = extractSinglesPairsMus(intersect_mus, region_mus, glp_state, nVals, pair_inds);
        end
        
        % Add standard spanning tree inequalities:
        if (strcmp(treeInequalityMode, 'standard'))
            % Find a MAXIMAL-weight spanning tree over the graph:
            [tree_edges, tree_pair_inds, tree_weight, tree_num_conn_comp] = ...
                calcMaxSpanningTree(x_n, pair_inds, all_mus, pair_mus, single_mus);
            
            if (~isempty(all_mus))
                if (DEBUG)
                    fprintf('Max spanning tree violation = %g\n', tree_weight);
                end
                if (tree_weight <= (tree_num_conn_comp - 1 + APPROX_ZERO_THRESH))
                    errMsg = 'All spanning tree inequalities are already satisfied :(';
                    disp(errMsg);
                    
                    if (TRY_DUAL_CLIQUE_TREE_INEQUALITIES)
                        treeInequalityMode = 'dual';
                        continue;
                    else
                        failed = true;
                        if (FALLBACK_ON_NILSSON_WHEN_FRACTIONAL)
                            disp('Fractional solution found - trying Nilsson''s method...');
                            NEXT_BEST_METH = 'Nilsson';
                        end
                        % cannot proceed, so just exit the loop and decode as best as possible:
                        break;
                    end
                end
            end
            
            if (~SAVE_MEMORY)
                added_trees(end+1,:) = tree_edges;
            end
            unique_trees = unique(added_trees, 'rows');
            numAddedTrees = size(added_trees,1);
            numUniqTrees = size(unique_trees,1);
            if (DEBUG)
                disp(['Adding spanning tree inequality # ', num2str(numAddedTrees), ' [edges = ', num2str(tree_edges), ']']);
            end
            if (numAddedTrees ~= numUniqTrees)
                error([num2str(numAddedTrees), ' = numAddedTrees ~= numUniqTrees = ', num2str(numUniqTrees)]);
            end
            [use_A(end+1,:), use_b(end+1,:)] = createTreeInequality(x_n, tree_pair_inds, tree_num_conn_comp, pair_start, single_start, numMuVars, nVals);
            numAddedIneq = numAddedIneq + 1;
            
            if (DEBUG && ~isempty(all_mus))
                treeViolation = use_A(end,:) * all_mus;
                if ((abs(treeViolation - tree_weight)) > APPROX_ZERO_THRESH)
                    error('Predicted maximal violation is not accurate!');
                end
                if (treeViolation <= (use_b(end,:) + APPROX_ZERO_THRESH))
                    error(['Spanning Tree Inequality already satisfied with value = ', num2str(treeViolation)]);
                end
            end
            
            % add "DUAL" spanning tree inequalities:
        elseif (strcmp(treeInequalityMode, 'dual'))
            if (isempty(dual_pair_inds))
                dual_pair_inds = createDualGraph(pair_inds);
            end
            
            % Find a MAXIMAL-weight spanning tree over the "DUAL" graph:
            [dual_tree_edges, dual_tree_pair_inds, dual_tree_weight, dual_tree_num_conn_comp] = ...
                calcMaxDualSpanningTree(x_n, pair_inds, dual_pair_inds, all_mus, pair_mus, single_mus);
            
            if (~isempty(all_mus))
                if (DEBUG)
                    fprintf('Max DUAL spanning tree violation = %g\n', dual_tree_weight);
                end
                if (dual_tree_weight <= (dual_tree_num_conn_comp - 1 + APPROX_ZERO_THRESH))
                    errMsg = 'All DUAL spanning tree inequalities are already satisfied :(';
                    disp(errMsg);
                    
                    if (TRY_REGION_SEP_TREE_INEQUALITIES)
                        treeInequalityMode = 'general_region';
                        continue;
                    else
                        failed = true;
                        if (FALLBACK_ON_NILSSON_WHEN_FRACTIONAL)
                            disp('Fractional solution found - trying Nilsson''s method...');
                            NEXT_BEST_METH = 'Nilsson';
                        end
                        % cannot proceed, so just exit the loop and decode as best as possible:
                        break;
                    end
                end
            end
            
            if (~SAVE_MEMORY)
                added_dual_trees(end+1,:) = dual_tree_edges;
            end
            unique_dual_trees = unique(added_dual_trees, 'rows');
            numAddedDualTrees = size(added_dual_trees,1);
            numUniqDualTrees = size(unique_dual_trees,1);
            if (DEBUG)
                disp(['Adding DUAL spanning tree inequality # ', num2str(numAddedDualTrees), ' [edges = ', num2str(dual_tree_edges), ']']);
            end
            if (numAddedDualTrees ~= numUniqDualTrees)
                error([num2str(numAddedDualTrees), ' = numAddedDualTrees ~= numUniqDualTrees = ', num2str(numUniqDualTrees)]);
            end
            
            [use_A(end+1,:), use_b(end+1,:)] = ...
                createDualTreeInequality(x_n, dual_tree_pair_inds, dual_tree_num_conn_comp, pair_inds, pair_start, single_start, numMuVars, nVals);
            numAddedIneq = numAddedIneq + 1;
            
            if (DEBUG && ~isempty(all_mus))
                dualTreeViolation = use_A(end,:) * all_mus;
                if ((abs(dualTreeViolation - dual_tree_weight)) > APPROX_ZERO_THRESH)
                    error('Predicted maximal DUAL violation is not accurate!');
                end
                if (dualTreeViolation <= (use_b(end,:) + APPROX_ZERO_THRESH))
                    error(['DUAL Spanning Tree Inequality already satisfied with value = ', num2str(dualTreeViolation)]);
                end
            end
            
            % add generalized region-separator spanning tree inequalities:
        elseif (strcmp(treeInequalityMode, 'general_region'))
            if (isempty(region_sep_pair_inds))
                region_sep_pair_inds = createRegionSeparatorGraph(glp_state);
            end
            
            % Find a MAXIMAL-weight spanning tree over the "REGIONS" graph:
            [use_regions, use_separators, separator_degs, region_sep_tree_weight, region_sep_tree_num_conn_comp] = ...
                calcMaxRegionsSpanningTree(x_n, glp_state, region_sep_pair_inds, nVals, all_mus, region_mus, intersect_mus);
            
            if (~isempty(all_mus))
                if (DEBUG)
                    fprintf('Max REGIONS spanning tree violation = %g\n', region_sep_tree_weight);
                end
                if (region_sep_tree_weight <= (region_sep_tree_num_conn_comp - 1 + APPROX_ZERO_THRESH))
                    errMsg = 'All REGIONS spanning tree inequalities are already satisfied :(';
                    disp(errMsg);
                    
                    failed = true;
                    if (FALLBACK_ON_NILSSON_WHEN_FRACTIONAL)
                        disp('Fractional solution found - trying Nilsson''s method...');
                        NEXT_BEST_METH = 'Nilsson';
                    end
                    % cannot proceed, so just exit the loop and decode as best as possible:
                    break;
                end
            end
            
            sepsUnionRegions = [use_separators, -1, use_regions, -1, separator_degs];
            sz = length(sepsUnionRegions);
            if (~SAVE_MEMORY)
                if (length(added_region_trees) < sz)
                    added_region_trees{sz} = [];
                end
                added_region_trees{sz}(end+1,:) = sepsUnionRegions;
            end
            unique_region_trees = unique(added_region_trees{sz}, 'rows');
            numAddedRegionTrees = size(added_region_trees{sz},1);
            numUniqRegionTrees = size(unique_region_trees,1);
            if (DEBUG)
                disp(['Adding REGIONS spanning tree inequality # ', num2str(numAddedRegionTrees), ...
                    ' [separators = ', num2str(use_separators), ', degrees = ', num2str(separator_degs), ', regions = ', num2str(use_regions), ']']);
            end
            if (numAddedRegionTrees ~= numUniqRegionTrees)
                error([num2str(numAddedRegionTrees), ' = numAddedRegionTrees ~= numUniqRegionTrees = ', num2str(numUniqRegionTrees)]);
            end
            
            [use_A(end+1,:), use_b(end+1,:)] = ...
                createRegionSeparatorTreeInequality(x_n, use_regions, use_separators, separator_degs, region_sep_tree_num_conn_comp, ...
                glp_state, region_start, intersect_start, numMuVars, nVals);
            numAddedIneq = numAddedIneq + 1;
            
            if (DEBUG && ~isempty(all_mus))
                regionTreeViolation = use_A(end,:) * all_mus;
                if ((abs(regionTreeViolation - region_sep_tree_weight)) > APPROX_ZERO_THRESH)
                    error('Predicted maximal REGIONS violation is not accurate!');
                end
                if (regionTreeViolation <= (use_b(end,:) + APPROX_ZERO_THRESH))
                    error(['REGIONS Spanning Tree Inequality already satisfied with value = ', num2str(regionTreeViolation)]);
                end
            end
        end
        
        % Solve the LP relaxation using the added inequalities:
        [region_mus, intersect_mus, enVal_ub, all_mus] = solve_glp_relax(glp_state, use_A, use_b, numAddedIneq, obj);
        
        % perform the MAP decoding:
        [x, enVal, fractional] = decodeMus(enVal_ub, all_mus, intersect_mus, region_mus, nVals, '', glp_state, pair_inds, lambda, local, DEBUG, x_n);
    end
    
    if (DEBUG && ~failed)
        disp(['Successfully added ', num2str(numAddedIneq), ' TOTAL inequalities']);
    end
    
    if (fractional)
        disp(['STRIPES iteration', ' solution is fractional :(']);
    end
    
    if (isempty(find(x ~= x_n, 1))) % x == x_n
        disp('x == x_n: MUST try Nilsson''s method...');
        NEXT_BEST_METH = 'Nilsson';
    end
    
    % Save the STRIPES result (so that the maximum can be taken if also running Nilsson):
    [STRIPES_x, STRIPES_enVal] = deal(x, enVal);
    
    stats.num_tree_ineqs = numAddedIneq;
    stats.fractional = failed;
end

if (strcmp(NEXT_BEST_METH, 'Nilsson'))
    [x, enVal] = deal(-ones(1,N), -inf);
    
    % Consider N partitions (of each variable having a value different than x_n):
    use_A = [A; space_n_lock_constraints_eqs];
    use_b = [b; space_n_lock_constraints_eq_consts];
    
    for numPositiveLockedVars = 0:N-1
        % Positively lock the prefix set of variables to be the same as x_n:
        if (numPositiveLockedVars > 0)
            % Change the negative lock (0) to a positive lock (1):
            use_b(end,:) = POS_LOCK;
            
            posLockVar = numPositiveLockedVars;
            posLockVal = x_n(posLockVar);
            if (DEBUG)
                if ((sum(issetpos{posLockVar}) > 0 && ~issetpos{posLockVar}(posLockVal)) ...
                        || issetneg{posLockVar}(posLockVal))
                    error(['LOGICAL error -- should not happen (since x_n is ALWAYS valid in space n):', ...
                        ' trying to positively lock a variable that is already positively locked to a DIFFERENT value,', ...
                        ' or it is negatively locked against this value!']);
                end
            end
        end
        
        % Negatively lock the last variable to be different than x_n:
        negLockVar = numPositiveLockedVars + 1;
        negLockVal = x_n(negLockVar);
        
        [use_A(end+1,:), use_b(end+1,:)] = ...
            createLockEqualities(negLockVar, negLockVal, NEG_LOCK, single_start, numMuVars);
        
        if (issetpos{negLockVar}(negLockVal) ... % pre-existing positive lock contradicts the negative lock
                ... % would lead to all values being negatively constrained:
                || (sum(~issetneg{negLockVar}) == 1 && ~issetneg{negLockVar}(negLockVal)))
            % added the lock equality above (for the sake of the next rounds), but no need to run LP (since infeasible):
            continue;
        end
        
        if (DEBUG)
            disp('Nilsson partition:');
            for i=1:numPositiveLockedVars
                fprintf('x_%i = %i, ', i, x_n(i));
            end
            fprintf('x_%i != %i\n', negLockVar, x_n(negLockVar));
        end
        
        [region_mus, intersect_mus, subspace_enVal_ub, all_mus] = solve_glp_relax(glp_state, use_A, use_b, 0, obj);
        if (DEBUG)
            disp('Nilsson partition result calculated.');
        end
        
        % perform the MAP decoding:
        [subspace_x, subspace_enVal] = decodeMus(subspace_enVal_ub, all_mus, intersect_mus, region_mus, nVals, 'Nilsson partition', glp_state, pair_inds, lambda, local, DEBUG);
        if (subspace_enVal > enVal)
            enVal = subspace_enVal;
            x = subspace_x;
        end
    end
    
    if (exist('STRIPES_enVal', 'var') && STRIPES_enVal > enVal)
        enVal = STRIPES_enVal;
        x = STRIPES_x;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (strcmp(NEXT_BEST_METH, 'MMN'))
    [x, enVal] = deal(-ones(1,N), -inf);
    high = 1e6;
    
    orig_local = glp_state.orig_local;
    for c=1:num_lock_constraints
        constraint = space_n_lock_constraints(c,:);
        [pos_neg, var, val] = deal(constraint(1), constraint(2), constraint(3));
        if (DEBUG)
            isNegChar = '';
            if (pos_neg == NEG_LOCK)
                isNegChar = '!';
            end
            fprintf('x_%i %s= %i\n', var, isNegChar, val);
        end
        if (pos_neg)
            nv = length(orig_local{var});
            % Block all other assignments
            orig_local{var}(setdiff([1:nv],val)) = -high;
        else
            % Block the blocked assignment
            orig_local{var}(val) = -high;
        end
    end
    
    % Consider N partitions (of each variable having a value different than x_n):
    curr_msgs = [];
    mplp_state = [];
    for numPositiveLockedVars = 0:N-1
        curr_local = orig_local;
        % Positively lock the prefix set of variables to be the same as x_n:
        if (numPositiveLockedVars > 0)
            % Change the negative lock (0) to a positive lock (1):
            posLockVar = numPositiveLockedVars;
            posLockVal = x_n(posLockVar);
            % Do a positive lock on this variable
            nv = length(orig_local{posLockVar});
            curr_local{posLockVar}(setdiff([1:nv],posLockVal))=-high;
        end
        
        % Negatively lock the last variable to be different than x_n:
        
        negLockVar = numPositiveLockedVars + 1;
        negLockVal = x_n(negLockVar);
        curr_local{negLockVar}(negLockVal) = -high;
        
        if (issetpos{negLockVar}(negLockVal) ... % pre-existing positive lock contradicts the negative lock
                ... % would lead to all values being negatively constrained:
                || (sum(~issetneg{negLockVar}) == 1 && ~issetneg{negLockVar}(negLockVal)))
            % added the lock equality above (for the sake of the next rounds), but no need to run LP (since infeasible):
            continue;
        end
        
        %        [assign,dual_obj_hist,int_val_hist,curr_msgs] = mplp_run_fromc_regions(glp_state.regions,glp_state.lambda,curr_local);
        [assign,int_val_hist,dual_obj_hist,curr_msgs,mplp_state] = mplp_run_mex(glp_state.regions,glp_state.lambda,curr_local,[],mplp_state);
        %                                                        mplp_mex(glp_state.regions,intersects,my_region_lambda,var_sizes,gmplp_state.region_subsets,msgs);
        curr_val = int_val_hist(end);
        if curr_val>enVal
            x = assign;
            enVal = curr_val;
        end
    end
    if (exist('STRIPES_enVal', 'var') && STRIPES_enVal > enVal)
        enVal = STRIPES_enVal;
        x = STRIPES_x;
    end
end






next_best_solutions_n = x;
next_best_energies_n = enVal;

% Update next_best_vars_vals:
vars_and_pairs_diff = find(x ~= x_n, 1); % find first such variable
if (isempty(vars_and_pairs_diff))
    error('Despite constraining to the contrary, x == x_n');
end
var_that_is_diff = vars_and_pairs_diff(1);

next_best_vars_vals_n = [var_that_is_diff x(var_that_is_diff)];







% Decode the single_mus into an assignment of the variables:
function [assign] = decodeSingleMus(single_mus, eliminatedAssign, VERBOSE)

if (nargin < 2)
    eliminatedAssign = [];
end
if (nargin < 3)
    VERBOSE = true;
end

N = length(single_mus);
assign = zeros(1,N);
for i=1:N
    [vv, assign(i)] = max(single_mus{i});
end
assign = ensureSingleVariableIsEliminated(eliminatedAssign, single_mus, assign, VERBOSE);




% Decode the pair_mus into an assignment of the variables:
function [assign] = decodePairMus(single_mus, pair_mus, adj, local, eliminatedAssign, VERBOSE)

if (nargin < 5)
    eliminatedAssign = [];
end
if (nargin < 6)
    VERBOSE = true;
end

[row_inds, col_inds] = find(triu(adj));
nEdges = length(row_inds);

N = length(local);
assign = zeros(1,N);

neighbs = {};
for i=1:N
    nVals(i) = length(local{i});
    neighbs{i} = find(adj(i,:));
    if (length(neighbs(i)) == 0)
        % assign the variables that have no edges:
        [vv, assign(i)] = max(single_mus{i});
    end
end

% CODE BELOW WILL NOT WORK [WILL GET STUCK IN INFINITE LOOP]
% IF HAVE MANY CONNECTED COMPONENTS (EACH WITH MORE THAN ONE NODE).
% IN THAT CASE, WE NEED TO KNOW IF ei = [ri,ci] IS THE FIRST EDGE WITHIN ITS
% CONNECTED COMPONENT.
firstEdge = true;
[assign, firstEdge] = ensureSingleVariableIsEliminated(eliminatedAssign, single_mus, assign, VERBOSE);

while (~isempty(find(~assign))) % there are still some unassigned variables
    for ei = 1:nEdges
        ri = row_inds(ei);
        ci = col_inds(ei);
        
        if ((~assign(ri) && ~assign(ci) ... % neither variable is assigned
                ... % and a non-ci neighbor of ri exists (or vice versa) and they could be assigned already:
                && (length(adj(ri,:)) > 1 || length(adj(ci,:)) > 1) && ~firstEdge) ...
                ... % both variables are already assigned:
                || (assign(ri) && assign(ci)))
            continue;
        end
        firstEdge = false;
        
        pair_mat = pair_mus{ri,ci};
        if (assign(ri))
            rVal = assign(ri);
            pair_mat = pair_mat(rVal, :);
        end
        if (assign(ci))
            cVal = assign(ci);
            pair_mat = pair_mat(:, cVal);
        end
        [maxValsRow, maxRowInd] = max(pair_mat, [], 1);
        [maxVal, maxColInd] = max(maxValsRow, [], 2);
        maxRowInd = maxRowInd(maxColInd);
        
        if (~assign(ri))
            assign(ri) = maxRowInd;
        end
        if (~assign(ci))
            assign(ci) = maxColInd;
        end
    end
end



% Decode the mus into an assignment of the variables:
function [assign, enVal, fractional] = decodeMus(enVal_ub, all_mus, intersect_mus, region_mus, nVals, PRINT_NAME, glp_state, adj, lambda, local, DEBUG, eliminatedAssign)

if (nargin < 12)
    eliminatedAssign = [];
end

VERBOSE = (~isempty(PRINT_NAME));

[single_mus, pair_mus] = extractSinglesPairsMus(intersect_mus, region_mus, glp_state, nVals, adj);

if (~isFractional(all_mus)) % The decoding is unique (and simple):
    assign = decodeSingleMus(single_mus);
    enVal = calcEnergy(assign, adj, lambda, local);
    fractional = false;
    return;
end
fractional = true;

% Take the better of the single and pairwise decodings:
singlesAssign = decodeSingleMus(single_mus, eliminatedAssign, VERBOSE);
singlesEnVal = calcEnergy(singlesAssign, adj, lambda, local);

pairsAssign = decodePairMus(single_mus, pair_mus, adj, local, eliminatedAssign, VERBOSE);
pairsEnVal = calcEnergy(pairsAssign, adj, lambda, local);

if (singlesEnVal > pairsEnVal)
    assign = singlesAssign;
    enVal = singlesEnVal;
else
    assign = pairsAssign;
    enVal = pairsEnVal;
end

APPROX_ZERO_THRESH = 1e-5;
if (abs(enVal - enVal_ub) <= APPROX_ZERO_THRESH)
    % isFractional(all_mus) == true, BUT can decode to an integral assignment with the same energy
    % (i.e., the fractional vertex and the integral vertex are tied, due to ties between integral vertices):
    fractional = false;
else
    if (VERBOSE)
        disp([PRINT_NAME, ' solution is fractional :(']);
    end
end

if (DEBUG)
    numDiffVars = length(find(singlesAssign ~= pairsAssign));
    if (numDiffVars > 0)
        disp(['There are ', num2str(numDiffVars), ' variables that differ between singles and pair assignments']);
        singlesEnVal, pairsEnVal
    end
    enVal_ub, enVal
end



% Assign a single variable to be different than eliminatedAssign, using the
% maximizers of single_mus:
function [assign, failure] = ensureSingleVariableIsEliminated(eliminatedAssign, single_mus, assign, VERBOSE)

APPROX_ZERO_THRESH = 1e-5;
N = length(single_mus);
failure = true;

% Assign a single variable to be DIFFERENT than eliminatedAssign:
if (~isempty(eliminatedAssign))
    for i=1:N
        maxVal = max(single_mus{i});
        maximizers = find(single_mus{i} >= maxVal - APPROX_ZERO_THRESH);
        validMaximizers = maximizers(find(maximizers ~= eliminatedAssign(i),1));
        if (~isempty(validMaximizers))
            assign(i) = validMaximizers(1);
            failure = false;
            break;
        end
    end
    if (failure)
        if (VERBOSE)
            disp('Could NOT find a tied variable with a non-eliminated maximizer value!');
            disp('Forcing a single variable to be non-maximizing (but non-eliminated)...');
        end
        for i=1:N
            nonZeroing = find(single_mus{i} > APPROX_ZERO_THRESH);
            validNonZeroing = nonZeroing(find(nonZeroing ~= eliminatedAssign(i),1));
            if (~isempty(validNonZeroing))
                assign(i) = validNonZeroing(1);
                failure = false;
                break;
            end
        end
        if (failure)
            error('Could not find a non-ZERO pseudo-marginal for any non-eliminated value!');
        end
    end
end



% Create the linear equality to enforce a lock
% pos_neg == 1  ->  var == val
% pos_neg == 0  ->  var != val
%
% NOTE: also works for locking MULTIPLE vars, where vars, vals, and pos_negs are vectors.
function [lock_eqs, lock_eq_consts] = createLockEqualities(vars, vals, pos_negs, single_start, numMuVars)

numVarsToLock = length(vars);
lock_eqs = sparse(numVarsToLock, numMuVars);
lock_eq_consts = zeros(numVarsToLock, 1);

muIndsToLock = single_start(vars) - 1 + vals;
for v = 1:numVarsToLock
    lock_eqs(v, muIndsToLock(v)) = 1;
    lock_eq_consts(v) = pos_negs(v);
end



% Create a sparse description for a given lock constraint:
function [descript] = createLockConstraint(pos_neg, var, val, single_start, numMuVars)

descript = [pos_neg, var, val];



% Checks that pair_inds contains the edge indices in increasing order starting
% from 1, and they are ordered by:
% [row_inds, col_inds] = find(triu(pair_inds))
function [] = requirePairIndsIsValidOrExit(pair_inds)

expected_ind = 1;

[row_inds, col_inds] = find(triu(pair_inds));
for ei = 1:length(row_inds)
    var1 = row_inds(ei);
    var2 = col_inds(ei);
    if (pair_inds(var1,var2) ~= expected_ind)
        error('The pair_inds matrix given is NOT VALID!');
    end
    
    expected_ind = expected_ind + 1;
end






% Finds the maximal spanning tree over the edge weights for the dual clique tree inequalities:
function [dual_tree_edges, dual_tree_pair_inds, dual_tree_weight, dual_tree_num_conn_comp] = ...
    calcMaxDualSpanningTree(elim_assign, pair_inds, dual_pair_inds, all_mus, pair_mus, single_mus)

N = length(pair_inds);
dual_pair_weights = sparse(size(dual_pair_inds,1), size(dual_pair_inds,2));
const_contrib = 0;

if (~isempty(all_mus))
    % Iterate over the separators in the dual graph:
    for i = 1:N
        edgeNeighbs = find(dual_pair_inds(i,:));
        edgeWeight = - single_mus{i}(elim_assign(i));
        dual_pair_weights(i, edgeNeighbs) = edgeWeight;
        dual_pair_weights(edgeNeighbs, i) = edgeWeight;
    end
    
    % Add the constant contributions of the separators:
    for i = 1:N
        const_contrib = const_contrib + single_mus{i}(elim_assign(i));
    end
    
    % Add the constant contributions of the cliques:
    [row_inds, col_inds] = find(triu(pair_inds));
    nEdges = length(row_inds);
    for ei = 1:nEdges
        var1 = row_inds(ei);
        var2 = col_inds(ei);
        [val1, val2] = deal(elim_assign(var1), elim_assign(var2));
        const_contrib = const_contrib + pair_mus{var1,var2}(val1, val2);
    end
end

% Find a MAXIMAL-weight spanning tree of the "dual" graph:
[dual_tree_adj, dual_tree_edges, dual_tree_pair_inds, dual_tree_weight, dual_tree_num_conn_comp] = ...
    UndirectedMaximumSpanningTree(dual_pair_inds, dual_pair_weights);
dual_tree_weight = dual_tree_weight + const_contrib;


% Finds the maximal spanning tree over the edge weights for the tree
% (of ALL regions and ALL separators) inequalities:
function [use_regions, use_separators, separator_degs, region_sep_tree_weight, region_sep_tree_num_conn_comp] = ...
    calcMaxRegionsSpanningTree(elim_assign, glp_state, region_sep_pair_inds, nVals, all_mus, region_mus, intersect_mus)

region_sep_pair_weights = sparse(size(region_sep_pair_inds,1), size(region_sep_pair_inds,2));
numSeps = length(glp_state.intersects);
numRegions = length(glp_state.regions);
const_contrib = 0;

if (~isempty(all_mus))
    % Iterate over the separators:
    for i = 1:numSeps
        interVars = glp_state.intersects{i};
        flat_ind = my_base2dec_multi(elim_assign(interVars) - 1, nVals(interVars)) + 1;
        elim_inter_mu_val = intersect_mus{i}(flat_ind);
        edgeWeight = - elim_inter_mu_val;
        
        neighbs = find(region_sep_pair_inds(i,:));
        region_sep_pair_weights(i, neighbs) = edgeWeight;
        region_sep_pair_weights(neighbs, i) = edgeWeight;
        
        % Add the constant contributions of the separator:
        const_contrib = const_contrib + elim_inter_mu_val;
    end
    
    % Add the constant contributions of the regions:
    for i=1:numRegions
        regVars = glp_state.regions{i};
        flat_ind = my_base2dec_multi(elim_assign(regVars) - 1, nVals(regVars)) + 1;
        elim_reg_mu_val = region_mus{i}(flat_ind);
        const_contrib = const_contrib + elim_reg_mu_val;
    end
end

% Find a MAXIMAL-weight spanning tree of the region-separator graph (where
% ALL regions and ALL separators are included):
[region_sep_tree_adj, region_sep_tree_edges, region_sep_tree_pair_inds, region_sep_tree_weight, region_sep_tree_num_conn_comp] = ...
    UndirectedMaximumSpanningTree(region_sep_pair_inds, region_sep_pair_weights);
region_sep_tree_weight = region_sep_tree_weight + const_contrib;

[allVarsInTree, dummy] = find(region_sep_tree_adj);
allVarsInTree = unique(allVarsInTree)';
use_separators = allVarsInTree(find(allVarsInTree <= numSeps));
use_regions = allVarsInTree(find(allVarsInTree > numSeps)) - numSeps;

region_sep_tree_degrees = sum(region_sep_tree_adj, 2)';
separator_degs = region_sep_tree_degrees(use_separators);



% Create a tree inequality (against elim_assign) using the tree described by tree_pair_inds:
function [tree_ineq, tree_ineq_const] = ...
    createTreeInequality(elim_assign, tree_pair_inds, tree_num_conn_comp, pair_start, single_start, numMuVars, nVals)

numGraphVars = length(nVals);

tree_adjMatrix = (tree_pair_inds > 0);
tree_degrees = sum(tree_adjMatrix, 2);

tree_ineq = sparse(1, numMuVars);

% Iterate over all pairwise edges:
[tree_row_inds, tree_col_inds] = find(triu(tree_pair_inds));
for ei = 1:length(tree_row_inds)
    ri = tree_row_inds(ei);
    ci = tree_col_inds(ei);
    
    %ei_in_graph = find(row_inds==ri & col_inds==ci);
    % This calculation is VALID since requirePairIndsIsValidOrExit() checked the original pair_inds:
    ei_in_graph = tree_pair_inds(ri,ci);
    
    flat_ind = my_base2dec_multi(elim_assign([ri ci]) - 1, nVals([ri ci])) + 1;
    useInd = pair_start(ei_in_graph) - 1 + flat_ind;
    tree_ineq(useInd) = 1;
end

% Iterate over all single variables:
for i = 1:numGraphVars
    tree_ineq(single_start(i) - 1 + elim_assign(i)) = 1 - tree_degrees(i);
end

tree_ineq_const = tree_num_conn_comp - 1;



% Create a dual tree inequality (against elim_assign) using the dual tree described by dual_tree_pair_inds:
function [dual_tree_ineq, dual_tree_ineq_const] = ...
    createDualTreeInequality(elim_assign, dual_tree_pair_inds, dual_tree_num_conn_comp, pair_inds, pair_start, single_start, numMuVars, nVals)

numGraphVars = length(nVals);

dual_tree_adjMatrix = (dual_tree_pair_inds > 0);
dual_tree_degrees = sum(dual_tree_adjMatrix, 2);

dual_tree_ineq = sparse(1, numMuVars);

% Iterate over all separators in the dual graph:
for i = 1:numGraphVars
    dual_tree_ineq(single_start(i) - 1 + elim_assign(i)) = 1 - dual_tree_degrees(i);
end

% Iterate over all cliques in the dual graph:
[row_inds, col_inds] = find(triu(pair_inds));
for ei = 1:length(row_inds)
    ri = row_inds(ei);
    ci = col_inds(ei);
    
    % This calculation is VALID since requirePairIndsIsValidOrExit() checked the original pair_inds:
    ei_in_graph = pair_inds(ri,ci);
    
    flat_ind = my_base2dec_multi(elim_assign([ri ci]) - 1, nVals([ri ci])) + 1;
    useInd = pair_start(ei_in_graph) - 1 + flat_ind;
    dual_tree_ineq(useInd) = 1;
end

dual_tree_ineq_const = dual_tree_num_conn_comp - 1;


% Create a region-separator tree graph inequality (against elim_assign) using
% the specified regions and their separators:
function [region_ineq, region_ineq_const] = ...
    createRegionSeparatorTreeInequality(elim_assign, use_regions, use_separators, separator_degs, region_sep_tree_num_conn_comp, ...
    glp_state, region_start, intersect_start, numMuVars, nVals)
region_ineq = sparse(1, numMuVars);

numRegions = length(use_regions);
for i=1:numRegions
    reg_ind = use_regions(i);
    regVars = glp_state.regions{reg_ind};
    
    flat_ind = my_base2dec_multi(elim_assign(regVars) - 1, nVals(regVars)) + 1;
    useInd = region_start(reg_ind) - 1 + flat_ind;
    region_ineq(useInd) = 1;
end

numInters = length(use_separators);
for i=1:numInters
    inter_ind = use_separators(i);
    interVars = glp_state.intersects{inter_ind};
    
    flat_ind = my_base2dec_multi(elim_assign(interVars) - 1, nVals(interVars)) + 1;
    useInd = intersect_start(inter_ind) - 1 + flat_ind;
    region_ineq(useInd) = 1 - separator_degs(i);
end

region_ineq_const = region_sep_tree_num_conn_comp - 1;
