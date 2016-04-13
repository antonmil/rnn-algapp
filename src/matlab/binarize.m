function ret = binarize(vec)
%% make sure vector only contains binary values
% i.e. clamp to [0,1] and round

vec(vec<0)=0; vec(vec>1)=1; vec=round(vec);
ret = vec;