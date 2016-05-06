function Marginals = Efficient_JPDA(Assign_matrix)
[MS1,MS] = size(Assign_matrix);
Marginals = zeros(MS1,MS);
if MS1~=MS
    error('not square matrix')
end
Hyp = perms(1:MS);
for j = 1: size(Hyp,1)
    lind= sub2ind(size(Assign_matrix),1:MS1,Hyp(j,:));
    Marginals(lind) = Marginals(lind) + prod(Assign_matrix(lind));
end