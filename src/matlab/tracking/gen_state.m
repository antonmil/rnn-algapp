function X= gen_state(F,Q,X0,K)
%--- this new state generation function follows the linear state spc eqn.
% x= Ax_old + Bv
X=cell(1,K);
B = chol(Q);
X{1}=X0';
for k=2:K
    X{k}= ((F*X{k-1})'+ randn(size(X{k-1},2),size(F,2))*B)';
end