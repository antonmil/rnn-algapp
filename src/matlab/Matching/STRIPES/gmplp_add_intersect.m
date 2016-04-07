function [gmplp_state,si] = gmplp_add_intersect(gmplp_state,new_intersect,new_intersect_lambda) %; %,lp_pair_bel)

if (nargin < 3)
  new_intersect_lambda = [];
end

% Sort region (we need this for the intersections)
new_intersect = sort(new_intersect);

sum_into_subsets = gmplp_state.sum_into_subsets;
intersect_with = gmplp_state.intersect_with;
region_subsets = gmplp_state.region_subsets;
regions = gmplp_state.regions;
intersects = gmplp_state.intersects;
intersect_lambda = gmplp_state.intersect_lambda;
curr_msgs = gmplp_state.curr_msgs;
mask = gmplp_state.mask;
mask_for_max = gmplp_state.mask_for_max;
region_id = gmplp_state.region_id;
intersect_id = gmplp_state.intersect_id;
intersect_size = gmplp_state.intersect_size;
var_size = gmplp_state.var_size;

intersects{end+1} = new_intersect;
intersect_lambda{end+1} = new_intersect_lambda;

intersect_id(end+1,new_intersect) = 1;
si = size(intersect_id,1);
intersect_size(si) = prod(var_size(new_intersect));
% Initialize the set of regions that we intersect with
intersect_with{si} = [];

intersect_lens = sum(intersect_id,2);
% Find the regions that intersect with this intersection
leninters = intersect_id(end,:)*region_id';   
% Consider the regions where the overlap is the size of the new intersection set
inters_withsi = find(leninters==length(new_intersect));
if ~isempty(inters_withsi)
    disp(['gmplp_add_intersect: Supporting addition of intersections only when they do not intersect with anything...']);
    return;
end
sum_into_subsets{si} = zeros(intersect_size(si),1);

% Note we do not need to change sum_into_subsets since the new messages are
% all zeros

gmplp_state.intersect_lambda = intersect_lambda;
gmplp_state.region_subsets = region_subsets;
gmplp_state.intersect_id = intersect_id;
gmplp_state.intersects = intersects;
gmplp_state.sum_into_subsets = sum_into_subsets;
%gmplp_state.curr_msgs = curr_msgs;
%gmplp_state.mask = mask;
%gmplp_state.mask_for_max = mask_for_max;
%gmplp_state.inds_in_region = inds_in_region;
gmplp_state.intersect_size = intersect_size;
gmplp_state.intersect_with = intersect_with;


return;

if 0
    for i=1:length(inters_withsi)
        % Get the intersecting region
        ri = inters_withsi(i);
        % Add it
        intersect_with{si}(end+1) = ri;
        region_subsets{ri}(end+1) = si;
        tmptmp=[];
        % Get the indices of the new intersection set within the region
        for ii=1:length(intersects{si})
            jj = find(intersects{si}(ii)==regions{ri});
            tmptmp(end+1) = jj;
        end
        inds_in_region{ri,length(region_subsets{ri})} = tmptmp;

        mask{ri,length(region_subsets{ri})} = get_marginal_mask_multi1(var_size(regions{ri}),length(regions{ri}),inds_in_region{ri,length(region_subsets{ri})});
        tt = mask{ri,length(region_subsets{ri})} ;
        tt(find(tt==0))=NaN;
        mask_for_max{ri,length(region_subsets{ri})} = tt;

        % Update the messages
    end


    % Initialize messages from the
    for sj=1:length(region_subsets{ri})
        curr_msgs{ri,sj} = zeros(intersect_size(region_subsets{ri}(sj)),1);
    end
end
