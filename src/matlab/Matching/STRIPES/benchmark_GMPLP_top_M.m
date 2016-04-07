%function [NEXT_BEST_METHS, MPLP_solutions, MPLP_energies, runTimes, NUM_RANKS] = ...
%    benchmark_GMPLP_top_M(pair_inds, lambda, local, glp_state, M, SAVE_NAME, DEBUG, RUN_BMMF, VERBOSE, RECOVERY_MODE, ONLY_BMMF, BENCHMARK_METHS, GBP_regions)
function [NEXT_BEST_METHS, MPLP_solutions, MPLP_energies, runTimes, NUM_RANKS] = ...
    benchmark_GMPLP_top_M(pair_inds, lambda, local, glp_state, M, SAVE_NAME, DEBUG, RUN_BMMF, VERBOSE, RECOVERY_MODE, ONLY_BMMF, BENCHMARK_METHS, GBP_regions)

if (nargin < 9)
  VERBOSE = true;
end
if (nargin < 10)
  RECOVERY_MODE = false;
end
if (nargin < 11)
  ONLY_BMMF = false;
end
if (nargin < 12)
  BENCHMARK_METHS = {};
end
if (nargin < 13)
  GBP_regions = {};
end

% Ensure that local is a row cell array:
if (size(local,1) > size(local,2))
  local = local';
end

if (RECOVERY_MODE && isempty(SAVE_NAME))
  disp('SAVE_NAME is empty, so cannot perform recovery mode');
  RECOVERY_MODE = false;
end

% The methods to benchmark:
NEXT_BEST_METHS = {};
if (ONLY_BMMF)
  if (~isempty(SAVE_NAME))
    SAVE_NAME = [SAVE_NAME, '.ONLY_BMMF'];
  end
else
  if (~isempty(BENCHMARK_METHS))
    NEXT_BEST_METHS = BENCHMARK_METHS;
  else % Use default methods:
    NEXT_BEST_METHS{end+1} = 'MMN';
    NEXT_BEST_METHS{end+1} = 'STRIPES';
    NEXT_BEST_METHS{end+1} = 'Nilsson';
  end
end
NUM_NEXT_BEST_METHS = length(NEXT_BEST_METHS);

MPLP_solutions = {};
MPLP_energies = {};
runTimes = [];

% Run the algorithms:
for k = 1:NUM_NEXT_BEST_METHS
  meth = NEXT_BEST_METHS{k};
  
  saveStatsName = '';
  if (~isempty(SAVE_NAME))
    saveStatsName = [SAVE_NAME, '.', meth];
  end
  
  USE_RECOVERY_MODE = RECOVERY_MODE;
  if (USE_RECOVERY_MODE)
    disp(['Attempting to recover the ', meth, ' method...']);
    
    recoverData = load('-mat', [saveStatsName, '.mat']);
    if (length(recoverData.MPLP_energies) == M)
      [MPLP_solutions{k}, MPLP_energies{k}, runTimes(k)] = ...
	  deal(recoverData.MPLP_solutions, recoverData.MPLP_energies, sum(recoverData.iterRunTimes));
    else
      USE_RECOVERY_MODE = false;
      disp(['Cached data file not valid for the ', meth, ' method!']);
    end
  end
  
  if (~USE_RECOVERY_MODE)
    disp(['Running the ', meth, ' method...']);
    
    [MPLP_solutions{k}, MPLP_energies{k}, iterRunTimes] = ...
	GMPLP_top_M(pair_inds, lambda, local, glp_state, M, meth, saveStatsName, DEBUG);
    runTimes(k) = sum(iterRunTimes);
  end
end

if (RUN_BMMF)
  NUM_VARS = length(local);
  NEXT_BEST_METHS{end+1} = 'BMMF';
  
  addpath(genpath('/cs/prt/fromer/cheny_scripts/Projects/Approximate Inference/approximate_inference'));
  addpath('/cs/prt/fromer/cheny_scripts/Projects/Approximate Inference/BMMF_c_inference/BMMF');
  
  potential_or_energy = 1;
  
  useLocals = cell(size(local));
  useLambda = sparse_cell(size(lambda,1),size(lambda,2));
  [x,y] = find(triu(pair_inds));

  if (potential_or_energy == 0)
    % Create potentials to be maximized [i.e. exponentiate energies to be maximized]:
    for i = 1:NUM_VARS
      useLocals{i} = exp(local{i});
    end
    for i=[x,y]'
      i1 = i(1); i2 = i(2);
      useLambda{i1,i2} = exp(lambda{i1,i2});
      useLambda{i2,i1} = useLambda{i1,i2}';
    end
  else % potential_or_energy == 1:
    % Create energies to be minimized [i.e. negate energies to be maximized]:
    for i = 1:NUM_VARS
      useLocals{i} = - local{i};
    end
    for i=[x,y]'
      i1 = i(1); i2 = i(2);
      useLambda{i1,i2} = - lambda{i1,i2};
      useLambda{i2,i1} = useLambda{i1,i2}';
    end
  end
  
  disp(['Running the BMMF method...']);
  startTime = cputime;
  [energy, res] = BMMF(pair_inds, useLocals, useLambda, M, potential_or_energy, GBP_regions, SAVE_NAME);
  endTime = cputime;
  runTimes(end+1) = endTime - startTime;

  % Since we want to maximize energy:
  energy = -energy';
  [MPLP_solutions{end+1}, MPLP_energies{end+1}] = deal(res, energy);
end



% Compare the algorithms:
[NUM_RANKS] = compare_runs(NEXT_BEST_METHS, MPLP_solutions, MPLP_energies, runTimes, M, SAVE_NAME, VERBOSE);
