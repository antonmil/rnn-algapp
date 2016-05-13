function [stateInfo,gtInfo] = reformat_Anton(XYZ,XeT,Ff,xy_GT)
Frame=size(XYZ,2);
N_T=size(XeT,2);

stateInfo.X=zeros(Frame,N_T);
stateInfo.Y=zeros(Frame,N_T);


for n=1:N_T
    stateInfo.X(Ff{n},n)=XeT{n}(1,:);
    stateInfo.Y(Ff{n},n)=XeT{n}(3,:);
end

stateInfo.frameNums=(1:Frame)-1;

if isempty(xy_GT)
    gtInfo =[];
else
    [~,Frame_GT,NT_GT]=size(xy_GT);
    
    gtInfo.X = zeros(Frame_GT,NT_GT);
    gtInfo.Y = zeros(Frame_GT,NT_GT);
    
    for n=1:NT_GT
        ix = (~isnan(xy_GT(1,:,n)));
        gtInfo.X(ix,n)=xy_GT(1,ix,n);
        gtInfo.Y(ix,n)=xy_GT(1,ix,n);
    end
    
    gtInfo.frameNums=(1:Frame)-1;
    gtInfo.Xgp=gtInfo.X; gtInfo.Ygp=gtInfo.Y;
end
stateInfo.Xgp=stateInfo.X; stateInfo.Ygp=stateInfo.Y;
