function  [trk1,trk2]=Track_Preprator(XeT,X1,Ff,Ff1,Frame)

N_Target=size(XeT,2);
trk2=zeros(size(XeT{1,1},1),Frame,N_Target);
for f=1:Frame
    for hh=1:N_Target
        indx=find(f==Ff{1,hh},1);
        if ~isempty(indx)
            trk2(:,f,hh)=XeT{1,hh}(:,indx);
        else
            trk2(:,f,hh)=NaN*ones(size(XeT{1,1},1),1);
        end
    end
end
N_Target2=size(X1,2);
trk1=zeros(size(X1{1,1},1),Frame,N_Target2);

for f=1:Frame
    for hh=1:N_Target2
        if f>=Ff1{1,hh}(1,1)&&f<=Ff1{1,hh}(1,2)
            trk1(:,f,hh)=X1{1,hh}(:,f-Ff1{1,hh}(1,1)+1);
        else
            trk1(:,f,hh)=NaN*ones(size(X1{1,1},1),1);
        end
    end   
end