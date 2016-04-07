function problem = makeProblem_hamid( M,nP1,nP2,GT)



E12 = ones(nP1,nP2);
[L12(:,1),L12(:,2)] = find(E12);
[group1,group2] = make_group12(L12);


%% Return results
problem.nP1 = nP1;
problem.nP2 = nP2;
problem.E12 = E12;
problem.L12 = L12;
problem.affinityMatrix = full(M);
problem.group1 = group1;
problem.group2 = group2;

problem.GTbool = GT(:);