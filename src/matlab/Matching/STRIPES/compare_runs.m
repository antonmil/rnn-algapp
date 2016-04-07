%function [NUM_RANKS] = compare_runs(NEXT_BEST_METHS, MPLP_solutions, MPLP_energies, runTimes, M, SAVE_NAME, VERBOSE)
function [NUM_RANKS] = compare_runs(NEXT_BEST_METHS, MPLP_solutions, MPLP_energies, runTimes, M, SAVE_NAME, VERBOSE)

if (nargin < 6)
  SAVE_NAME = '';
end
if (nargin < 7)
  VERBOSE = false;
end

% Graph constants:
AXIS_FONT_SIZE = 18;
MARKER_SIZE = 15;

BLACK = 'k';
GRAY = [211 211 211]/255;
CORNFLOWER_BLUE = [100 149 237]/255;
DARK_OLIVE_GREEN = [85 107 47]/255;
DARK_GOLDENROD = [184 134 11]/255;
INDIAN_RED = [205 92 92]/255;
DARK_VIOLET = [148 0 211]/255;
COLORS = ...
    {'b', 'g', 'r', 'c', 'm', BLACK, GRAY, ...
     CORNFLOWER_BLUE, DARK_OLIVE_GREEN, ...
     DARK_GOLDENROD, INDIAN_RED, DARK_VIOLET ...
    };


NUM_NEXT_BEST_METHS = length(NEXT_BEST_METHS);

if (VERBOSE && length(MPLP_solutions) >= 2)
  [only_A, only_B, same_A, same_B] = diffMatRows(MPLP_solutions{1}, MPLP_solutions{2});
  numSameSolutions_1_2 = length(same_A)
  absDiffMaxEns_1_2 = abs(max(MPLP_energies{1}) - max(MPLP_energies{2}))
  
  NEXT_BEST_METHS
  showRunTimes = num2str(runTimes)
end

% Check which algorithm did best:
ALL_SOLUTIONS = [];
for k=1:NUM_NEXT_BEST_METHS
  ALL_SOLUTIONS = unique([ALL_SOLUTIONS; MPLP_solutions{k}], 'rows');
end
NUM_ALL_SOLUTIONS = size(ALL_SOLUTIONS,1);
if (NUM_ALL_SOLUTIONS == 0)
  disp('Return: NUM_ALL_SOLUTIONS == 0');
  return;
end

NUM_TOP_RANKS = min(M, NUM_ALL_SOLUTIONS);

% Calculate the highest energy observed for each solution:
ALL_ENS = -inf * ones(size(ALL_SOLUTIONS,1),1);
for k = 1:NUM_NEXT_BEST_METHS
  methSol = MPLP_solutions{k};
  numMethSol = size(methSol,1);
  for m = 1:numMethSol
    sol = methSol(m,:);
    [only_A, only_B, same_A, same_B] = diffMatRows(sol, ALL_SOLUTIONS);
    if (length(same_B) ~= 1)
      error('Logical error [CANNOT BE SINCE: sol is ONE ROW, unique(ALL_SOLUTIONS) above!]');
    end
    ALL_ENS(same_B) = max(ALL_ENS(same_B), MPLP_energies{k}(m));
  end
end

TOP_ENS = sort(ALL_ENS, 1, 'descend');
TOP_ENS = TOP_ENS(1:NUM_TOP_RANKS);
M_LOWEST_RANKED_EN = TOP_ENS(end);

APPROX_ZERO_THRESH = 1e-5;
USE_M_LOWEST_RANKED_EN = M_LOWEST_RANKED_EN - APPROX_ZERO_THRESH;

NUM_RANKS = zeros(1,NUM_NEXT_BEST_METHS);
for k = 1:NUM_NEXT_BEST_METHS
  % Find the maximal energies for the solutions of the k-th algorithm:
  [only_A, only_B, same_A, same_B] = diffMatRows(MPLP_solutions{k}, ALL_SOLUTIONS);
  if (length(same_B) ~= size(MPLP_solutions{k},1))
    addRunName = '';
    if (~isempty(SAVE_NAME))
      addRunName = [' (', SAVE_NAME, ')'];
    end
    disp(['WARNING: MPLP_solutions for ', NEXT_BEST_METHS{k}, addRunName, ' has repeated solutions!']);
    same_B = unique(same_B);
  end
  USE_MPLP_energies_k = ALL_ENS(same_B);
  % Count the number of sequences with energy >= than the M-th lowest energy
  % (ACCOUNTING FOR POSSIBLE TIES, SO THAT RANK could THEORETICALLY be greater than M):
  NUM_RANKS(k) = sum(USE_MPLP_energies_k >= USE_M_LOWEST_RANKED_EN);
end

if (VERBOSE)
  NUM_RANKS
end


if (~isempty(SAVE_NAME))
  % Plot a figure comparing the energies:
  figure(1);
  clf
  hold on;
  for k = 1:NUM_NEXT_BEST_METHS
    methEns = MPLP_energies{k};
    color = COLORS{k};
    plot(1:length(methEns), methEns, '.', 'Color', color, 'MarkerSize', MARKER_SIZE);
  end
  set(gca,'FontSize',AXIS_FONT_SIZE);

  xlabel('Iteration #');
  ylabel('Energy');
  legend(NEXT_BEST_METHS, 'Location', 'NorthOutside');

  saveFig(gcf, [SAVE_NAME, '.method_compare']);
  
  % Store the results:
  save([SAVE_NAME, '.ALL_METHODS.mat'], 'NEXT_BEST_METHS', 'MPLP_solutions', 'MPLP_energies', 'runTimes', 'NUM_RANKS');
end
