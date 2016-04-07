%function [NEXT_BEST_METHS, MPLP_solutions, MPLP_energies, runTimes, NUM_RANKS] = ...
%    test_gnbest(NUM_VARS, k, M, graph_type, ADD_DEFAULT_REGIONS, seed, DEBUG, RUN_BMMF, SAVE_STATS, USE_MIN_CUT_BINARY_ENERGIES, BINARY_MAGNITUDE)
%
% Generate polytope for graph of graph_type with NUM_VARS variables, with
% randomized potentials.
%
% graph_type = 'chain', 'complete', 'grid', 'cycle', 'star', 'empty'
function [NEXT_BEST_METHS, MPLP_solutions, MPLP_energies, runTimes, NUM_RANKS] = ...
    test_gnbest(NUM_VARS, k, M, graph_type, ADD_DEFAULT_REGIONS, seed, DEBUG, RUN_BMMF, SAVE_STATS, USE_MIN_CUT_BINARY_ENERGIES, BINARY_MAGNITUDE)

if (nargin < 1)
    NUM_VARS = 10;
end
if (nargin < 2)
    k = 5; % variable arity
end
if (nargin < 3)
    M = 100; % number of top solutions
end
if (nargin < 4)
    graph_type = 'grid';
end
if (nargin < 5)
    ADD_DEFAULT_REGIONS = true;
end
if (nargin < 6)
    c = clock;
    seed = c(3);
    seed = c(4);
    seed = seed * 60 + c(5);
    seed = seed * 60 + c(6);
    seed = round(seed);
end
if (nargin < 7)
    DEBUG = false;
end
if (nargin < 8)
    RUN_BMMF = false;
end
if (nargin < 9)
    SAVE_STATS = false;
end
if (nargin < 10)
    USE_MIN_CUT_BINARY_ENERGIES = false;
end
if (nargin < 11)
    BINARY_MAGNITUDE = 0.5;
end

if (M > k^NUM_VARS)
    M = k^NUM_VARS;
    disp(['Resetting M to be = ', num2str(M)])
end

BINARY_MAGNITUDE_STRING = '';
if (k == 2)
    BINARY_MAGNITUDE_STRING = sprintf(', BINARY_MAGNITUDE = %s', num2str(BINARY_MAGNITUDE));
end

fprintf('# vars = %i, arity = %i, M = %i, graph_type = %s, ADD_DEFAULT_REGIONS = %i, seed = %i%s\n', ...
    NUM_VARS, k, M, graph_type, ADD_DEFAULT_REGIONS, seed, BINARY_MAGNITUDE_STRING);

rand('seed', seed);

% chain graph:
if (strcmp(graph_type,'chain'))
    adj = sparse(NUM_VARS, NUM_VARS);
    for i=1:NUM_VARS-1
        adj(i,i+1) = 1;
    end
    % complete graph:
elseif (strcmp(graph_type,'complete'))
    adj = ones(NUM_VARS) - eye(NUM_VARS);
    % grid graph:
elseif (strcmp(graph_type,'grid'))
    if (isprime(NUM_VARS))
        NUM_VARS = NUM_VARS + 1;
    end
    f = sort(factor(NUM_VARS));
    num_factors = length(f);
    if (mod(num_factors,2))
        f = [1 f];
    end
    num_factors = length(f);
    [nPerSideX, nPerSideY] = deal(1,1);
    for i=1:num_factors/2
        nPerSideX = nPerSideX * f(2*i - 1);
        nPerSideY = nPerSideY * f(2*i);
    end
    fprintf('Creating %i x %i grid graph.\n', nPerSideX, nPerSideY);
    
    edgeIndicator = [];
    for i=1:nPerSideX
        for j=1:nPerSideY
            edgeIndicator(end+1,:) = [i j];
        end
    end
    
    adj = sparse(NUM_VARS, NUM_VARS);
    for i=1:NUM_VARS
        for j=1:NUM_VARS
            if sum(abs(edgeIndicator(i,:)-edgeIndicator(j,:)))==1
                adj(i,j)=1;
            end
        end
    end
    adj = adj - diag(diag(adj));
    % cycle graph:
elseif (strcmp(graph_type,'cycle'))
    adj = sparse(NUM_VARS, NUM_VARS);
    for i=1:NUM_VARS-1
        adj(i,i+1) = 1;
    end
    adj(1,NUM_VARS) = 1;
    % star graph:
elseif (strcmp(graph_type,'star'))
    adj = sparse(NUM_VARS, NUM_VARS);
    for i=1:NUM_VARS-1
        adj(i,NUM_VARS) = 1;
    end
    % empty graph:
elseif (strcmp(graph_type,'empty'))
    adj = sparse(NUM_VARS, NUM_VARS);
else
    error(['Undefined graph_type = ', graph_type]);
end

% Create the indices matrix in the correct format:
[pair_inds, adj] = convertAdjToPairInds(adj);

% pair and local potentials for graph:
lambda = sparse_cell(NUM_VARS,NUM_VARS);
local = cell(1,NUM_VARS);

% Choose random potentials for the graph:
if (USE_MIN_CUT_BINARY_ENERGIES && k == 2)
    disp('Using ATTRACTIVE energies for binary cut graph...');
end

if (k == 2)
    DOUBLE_THE_MAGNITUDE = 2 * BINARY_MAGNITUDE; % To multiply range of +/- 0.5
end

for i=1:NUM_VARS
    local{i} = rand(k,1) - 0.5;
    
    if (k == 2)
        % Modify the local potentials accordingly:
        local{i} = DOUBLE_THE_MAGNITUDE *local{i};
        local{i}(1) = 0;
    end
    
    for j=i+1:NUM_VARS
        if ~pair_inds(i,j)
            continue;
        end
        
        if (k == 2)
            w = DOUBLE_THE_MAGNITUDE * (rand - 0.5);
            if (USE_MIN_CUT_BINARY_ENERGIES)
                w = abs(w);
            end
            lambda{i,j} = [w 0; 0 w];
        else % for k != 2, use standard "mixed" energies:
            lambda{i,j} = (rand(k,k) - 0.5);
        end
        
        lambda{j,i} = lambda{i,j}';
    end
end


% Create the glp_state and add regions as specified:
GBP_regions = {};
glp_state = [];

if (ADD_DEFAULT_REGIONS)
    disp('Adding DEFAULT regions (if defined)...');
    
    if (strcmp(graph_type,'grid'))
        USE_TRIANGLE_REGIONS = true;
        if (USE_TRIANGLE_REGIONS)
            disp('Adding TRIANGLE regions to grid.');
        else
            disp('Adding SQUARE regions to grid.');
        end
        [glp_state] = ginit_from_lambda(pair_inds, lambda, local, ...
            true); % ADD_ALL_EDGES_TO_INTERSECTS
        for x=1:nPerSideX-1
            for y=1:nPerSideY-1
                firstVar = y + (x - 1) * nPerSideY;
                secondVar = firstVar + 1;
                thirdVar = firstVar + nPerSideY;
                fourthVar = thirdVar + 1;
                
                if (USE_TRIANGLE_REGIONS)
                    triOverlapPair = [secondVar thirdVar];
                    glp_state = gmplp_add_intersect(glp_state, triOverlapPair);
                    triRegion1 = [firstVar secondVar thirdVar];
                    glp_state = gmplp_add_region(glp_state, triRegion1, []);
                    triRegion2 = [secondVar thirdVar fourthVar];
                    glp_state = gmplp_add_region(glp_state, triRegion2, []);
                    
                    GBP_regions{end+1} = triRegion1;
                    GBP_regions{end+1} = triRegion2;
                else % Use SQUARE regions:
                    squareRegion = [firstVar secondVar thirdVar fourthVar];
                    glp_state = gmplp_add_region(glp_state, squareRegion, []);
                    
                    GBP_regions{end+1} = squareRegion;
                end
            end
        end
    end
end

if (isempty(glp_state)) % for example, if ~ADD_DEFAULT_REGIONS
    disp('NO regions were added.');
    [glp_state] = ginit_from_lambda(pair_inds, lambda, local);
end


statsName = '';
if (SAVE_STATS)
    statsName = ['n_', num2str(NUM_VARS), '.k_', num2str(k), '.M_', num2str(M), ...
        '.graph_', graph_type, '.addRegions_', num2str(ADD_DEFAULT_REGIONS), ...
        '.seed_', num2str(seed)];
    if (USE_MIN_CUT_BINARY_ENERGIES && k == 2)
        statsName = [statsName, '.attractive_cut_energies'];
    end
end

fprintf('\n');

% Benchmark the algorithms:
[NEXT_BEST_METHS, MPLP_solutions, MPLP_energies, runTimes, NUM_RANKS] = ...
    benchmark_GMPLP_top_M(pair_inds, lambda, local, glp_state, M, statsName, DEBUG, RUN_BMMF, ~SAVE_STATS, ...
    false, false, {'STRIPES'}, GBP_regions);
