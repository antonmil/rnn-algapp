function [candidates, values, t] = CbasedBinIntMBest(f,A,b,Aeq,beq,options,M)
xdim = length(f);

candidates = zeros(xdim, M);
values = zeros(1, M);
t = zeros(1,M);

tstart = tic();
for i=1:M
    [y1, v1] = gurobi_ilp(f, A, b, Aeq, beq);
    t1 = toc(tstart);
    t(i) = t1;
    A = [A;y1'];
    b = [b; sum(y1(:)) - 1];
    if(~isempty(y1))
        candidates(:, i) = y1;
        values(i) = v1;
    else
        M = i - 1;
        break;
    end
end
t = t(1:M);
values = values(1:M);
candidates = candidates(:,1:M);
end