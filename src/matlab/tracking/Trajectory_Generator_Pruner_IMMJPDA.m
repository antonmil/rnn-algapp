function [X,F,hhat_X,hhat_tag]=Trajectory_Generator_Pruner_IMMJPDA(X,F,K,tr)


F(cellfun('size', X,2)<=tr)=[];
X(cellfun('size', X,2)<=tr)=[];


hhat_X=cell(1,K);
hhat_tag=cell(1,K);
for k=1:K
    for i=1:size(X,2)
        inx= find(F{i}==k);
        if ~isempty(inx)
            hhat_X{1,k} =[hhat_X{1,k} X{i}(:,inx)];
            hhat_tag{1,k} =[hhat_tag{1,k} i];
        end
    end
end

