function [Z,Iden]= gen_detection2(H,R,X,PD,range_c,lambda_c,tre)
% this observation generation function is for coordinate
% measurements
K=size(X,2);
z_dim=size(H,1);
Z=cell(1,K);
Iden=cell(1,K);
B = chol(R);

for k=1:K

    Xm=X{k};
        
    idx= find( rand(size(Xm,2),1) <= PD );
    if ~isempty(idx)
    Iden{k}=idx';
    Z{k}= ((H*Xm(:,idx))'+ randn(size(Xm(:,idx),2),z_dim)*B)';
    end
    N_c= poissrnd(lambda_c);    %no. of clutter points
    C= repmat(range_c(:,1),[1 N_c])+ diag(range_c*[ -1; 1 ])*rand(z_dim,N_c);  %clutter generation
    Z{k}= [ Z{k} C ];
    Iden{k}=[Iden{k} zeros(1,size(C,2))];
end
