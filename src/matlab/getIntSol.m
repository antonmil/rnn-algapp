function ass = getIntSol(mat)
% extracts N-dimensional integer vector from a binary
% assignment matrix


% e.g.
% 0 1 0
% 0 0 1
% 1 0 0
% should correspond to ass=[2,3,1]

[ass,~] = find(mat'); % transpose mat for row-wise assignment
ass=reshape(ass,1,length(ass)); % make row vector

end