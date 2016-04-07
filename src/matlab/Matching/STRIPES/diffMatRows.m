%function [only_A, only_B, same_A, same_B] = diffMatRows(A, B)
function [only_A, only_B, same_A, same_B] = diffMatRows(A, B)

THRESH = 1e-5;

only_A = [];
same_A = [];

only_B = [];
same_B = [];

if (size(A,2) ~= size(B,2))
  error('A and B must have the same number of columns!');
end

numRows_A = size(A,1);
numRows_B = size(B,1);
pair_wise_diffs = zeros(numRows_A, numRows_B);

% Calculate all pairwise differences between rows:
for row_ind_A = 1:numRows_A
  row_A = A(row_ind_A,:);
  
  for row_ind_B = 1:numRows_B
    row_B = B(row_ind_B,:);
    pair_wise_diffs(row_ind_A, row_ind_B) = sum((row_A - row_B).^2);
  end
end

% For each row of A, check if there exists an equal row in B:
for row_ind_A = 1:numRows_A
  [diff, row_ind_B] = min(pair_wise_diffs(row_ind_A, :));

  if diff < THRESH
    same_A(end+1) = row_ind_A;
  else
    only_A(end+1) = row_ind_A;
  end
end

% For each row of B, check if there exists an equal row in A:
for row_ind_B = 1:numRows_B
  [diff, row_ind_A] = min(pair_wise_diffs(:, row_ind_B));

  if diff < THRESH
    same_B(end+1) = row_ind_B;
  else
    only_B(end+1) = row_ind_B;
  end
end
