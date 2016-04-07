% Save figure as fig, color eps file, and optionally as tif
% h = figure handle
% function saveFig(h, filename, saveAsTiff)
function saveFig(h, filename, saveAsTiff)

if (nargin < 3)
  saveAsTiff = false;
end

% 1. Save as .eps:
print(h, '-depsc', [filename, '.eps']);

% 2. Save as .tif:
if (saveAsTiff)
  print(h, '-dtiff', [filename, '.tif']);
end

% 3. Save as .fig:
filenameFixedForSaveAs = replaceString(filename, '*', '_STAR_');
saveas(h, [filenameFixedForSaveAs, '.fig'], 'fig');

if (~strcmp(filename, filenameFixedForSaveAs))
  filenameFixedForMove = replaceString(filename, '*', '\*');
  mvCommand = ['mv -f ', filenameFixedForSaveAs, '.fig ', filenameFixedForMove, '.fig'];
  unix(mvCommand);
end
