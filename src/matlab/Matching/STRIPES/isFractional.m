function [has_fractional_values] = isFractional(data)

THRESH = 1e-5;
fractional_inds = find(abs(data - round(data)) > THRESH);
has_fractional_values = ~isempty(fractional_inds);
