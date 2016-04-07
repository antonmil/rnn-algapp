function [gmplp_state] = gmplp_add_region(gmplp_state,new_region,new_region_lambda) %; %,lp_pair_bel)

new_region = sort(new_region);

sum_into_subsets = gmplp_state.sum_into_subsets;
intersect_with = gmplp_state.intersect_with;
region_subsets = gmplp_state.region_subsets;
regions = gmplp_state.regions;
intersects = gmplp_state.intersects;
lambda = gmplp_state.lambda;
curr_msgs = gmplp_state.curr_msgs;
mask = gmplp_state.mask;
mask_for_max = gmplp_state.mask_for_max;
region_id = gmplp_state.region_id;
intersect_id = gmplp_state.intersect_id;
intersect_size = gmplp_state.intersect_size;
var_size = gmplp_state.var_size;
inds_in_region = gmplp_state.inds_in_region;

region_id(end+1,new_region) = 1;
lambda{end+1} = new_region_lambda;
regions{end+1} = new_region;
region_subsets{end+1} = [];
% Sort regions (we need this for the intersections)
intersect_lens = sum(intersect_id,2);
% Find the intersection sets for this new region
leninters = intersect_id*region_id(end,:)';   
inters_withri = find(leninters==intersect_lens);
if length(new_region)==3
  inters_withri = find(leninters==intersect_lens & leninters==2);
else
  inters_withri = find(leninters==intersect_lens);
end

ri = length(regions);

for i=1:length(inters_withri)
    si = inters_withri(i);
    intersect_with{si}(end+1) = ri;
    region_subsets{ri}(end+1) = si;
    tmptmp=[];
    for ii=1:length(intersects{si})
        jj = find(intersects{si}(ii)==regions{ri});
        tmptmp(end+1) = jj;
    end
    inds_in_region{ri,length(region_subsets{ri})} = tmptmp;

%    mask{ri,length(region_subsets{ri})} = get_marginal_mask_multi1(var_size(regions{ri}),length(regions{ri}),inds_in_region{ri,length(region_subsets{ri})});
%    tt = mask{ri,length(region_subsets{ri})} ;
%    tt(find(tt==0))=NaN;
%    mask_for_max{ri,length(region_subsets{ri})} = tt;
end

for sj=1:length(region_subsets{ri})
    curr_msgs{ri,sj} = zeros(intersect_size(region_subsets{ri}(sj)),1);
end
% Note we do not need to change sum_into_subsets since the new messages are
% all zeros

gmplp_state.intersect_with = intersect_with;
gmplp_state.region_subsets = region_subsets;
gmplp_state.regions = regions;
gmplp_state.lambda = lambda;
gmplp_state.region_id = region_id;
gmplp_state.curr_msgs = curr_msgs;
gmplp_state.mask = mask;
gmplp_state.mask_for_max = mask_for_max;
gmplp_state.inds_in_region = inds_in_region;