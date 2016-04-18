function [model, params] = getGurobiModel(N)
    M = N;
    A=zeros(M,N*M); Aeq=zeros(N, N*M);  % ineq and eq constr. matrices
    b=ones(M,1); beq=ones(N,1);         % ineq and eq constr. vectors
    act=1;
    for i=1:N
        Aeq(i,act:act+M-1)=1;
        act=act+M;
    end
    for j=1:M
        A(j,j:M:end)=1;
        act=act+M;
    end

    model.A = sparse([A; Aeq]);
    model.rhs = [b; beq];
    model.vtype = 'B'; % binary
    % model.vtype = 'C'; model.lb=0*ones(1,N*M); model.ub=1*ones(1,N*M); % relaxed
    model.modelsense = 'max';
    model.sense = char( ['<' * ones(1, length(b)), '=' * ones(1,length(beq))]);
    model.obj = ones(1,N*M);

    params.outputflag = 0;
    params.TimeLimit = .0005; % time limit in seconds
    
end