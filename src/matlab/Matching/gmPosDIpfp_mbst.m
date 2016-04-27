function [Xm,Objm,ttme]= gmPosDIpfp_mbst(K, Ct, X0, par)
% This function tries to find m-best solutions using naive approach for the
% following solver:
% The solver tries to maximize the matching score x' M x
% where x obeys discrete one-to-one matching constraints
% such that x(i) = 1 if nodes(i) is matched to labels(i) and 0 otherwise.
%
% Reference
%   M. Leordeanu and M. Hebert and R. Sukthankar, "An Integer Projected
%   Fixed Point Method for Graph Matching and MAP Inference", in NIPS, 2009
%
% Remark
%   nn = n1 x n2
%
% Input
%   K        -  affinity matrix, nn x nn (sparse)
%   Ct       -  constraints, n1 x n2
%   X0       -  initial assignment, n1 x n2
%   par      -  parameter
%     nItMa  -  #maximum iteration steps, {50}
%
% Output
%   X        -  m-permutation matrix, n1 x n2 x m
%
% History
%   create   -  Marius Leordeanu (leordeanu@gmail.com), 02-25-2011
%   modify   -  Seyed Hamid Rezatofighi(s.h.r.tofighi@gmail.com), 22-10-2015



siz1=size(Ct,1);siz2=size(Ct,2);
% ------------Inequaluity and Equality Contraint ------------------
bin2=ones(siz2,1);
Ain2=zeros(siz2,siz1*siz2);
beq=ones(siz1,1);
Aeq=zeros(siz1,siz1*siz2);
act=1;
for i=1:siz1
    Aeq(i,act:act+siz2-1)=1;
    act=act+siz2;
end
for j=1:siz2
    Ain2(j,j:siz2:end)=1;
    act=act+siz2;
end


mbst=par.mbst;
Xm = zeros(siz1,siz2, mbst);
Objm=zeros(1,mbst);
ttme=zeros(mbst,1);
sizA1=size(Ain2,1);
sizA2=size(Ain2,2);
Ain=zeros(sizA1+mbst,sizA2);
Ain(1:sizA1,:)=Ain2;
bin=zeros(sizA1+mbst,1);
bin(1:sizA1,:)=bin2;

for m=1:mbst
    tcc= tic;
    Xx = gmPosDIpfp2(K, Ct, X0, Ain(1:sizA1+m-1,:), bin(1:sizA1+m-1,:), Aeq, beq,par);
    X=abs(round(Xx));
    Xm(:,:,m) = X;
    if par.chck_sols
        if ~all(sum(X)<=ones(1,siz2))||~all(sum(X,2)<=ones(siz1,1))
            error('not one-to-one result')
        end
%         if ~all(all(X==Xx))
%             error('The result is not binary')
%         end
    end
    Objm(1,m)=X(:)'*K*X(:);
    Ain(sizA1+m,:)=X(:)';
    bin(sizA1+m,:)=sum(X(:))-1;
    ttme(m)=toc(tcc);
%     ttme(m)
end

    
if par.chck_sols
    Xres=Ain(sizA1+1:sizA1+mbst,:);
    if (size(unique(Xres,'rows'),1)~=size(Xres,1))
%         warning('not unique results')
    end
end



end
function x_opt=gmPosDIpfp2(K, Ct, X0, Ain, bin, Aeq, beq,par)
% function parameter
nItMa = ps(par, 'nItMa', 50);

% dimension
[n1, n2] = size(X0);

M = K;


%% initializations

% n = size(M,1);
% 
% nLabels = max(labels);
% nNodes  = max(nodes);

xc = X0(:);

sol = X0(:);

score_sol = 0;

score(1) =  xc'*M*xc;

%% climbing ------------------------------------------------------------------

for nSteps = 1:nItMa
          
   x0 = xc;
      
   xx =  M*x0;

%    A = -Inf*ones(nNodes, nLabels);
% 
%    for i = 1:nNodes
% 
%        f = find(nodes == i);
%        
%        A(i, labels(f)) = xx(f);
% 
%    end
%    
%    A = max(A(:))-A;
   Ax2=max(xx)-xx;
%    [X, s_aux]=hungarian(A);
   
   [b,~] = gurobi_ilp(Ax2, Ain, bin, Aeq, beq);
%    X=reshape(Xres,[nNodes,nLabels]);
   
%    b =  X;
   
%    for i = 1:nNodes
%       
%       f = find(nodes == i);
%       
%       match_ind = find(X(i,labels(f)) == 1);
%       
%       b(f(match_ind)) = 1;
%      
%    end
       
   dx = b - x0;
   
   A = dx'*M*dx;
  
   t = 1;
   
   if A < 0

       C = (x0'*M)*dx;
       
       t = min([1, -C/A]);
       
       if t < 0.01
           t = 0;
       end
   end
   
   xc = x0 + t*dx;
   
   score(nSteps+1) = xc'*M*xc;
     
   score_b = b'*M*b;
   
   if score_b >= score_sol
       
        score_sol = score_b;
        
        sol = b;
 
   end
      
   if norm(xc - x0) == 0
       break;
   end
   
end

x_opt = reshape(xc, [n1 n2]);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
