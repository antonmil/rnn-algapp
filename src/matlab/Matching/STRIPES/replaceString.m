%Function to replace fromStr to toStr in str
%function [repString] = replaceString(str, fromStr, toStr)
function [repString] = replaceString(str, fromStr, toStr)

% Update fromStr to be toStr:
fromStrLength = length(fromStr);
toStrLength = length(toStr);

fromIndices = strfind(str, fromStr);

repString = '';
prevIndex = 1;
for f = fromIndices
  % Insert the previous segment:
  for j = prevIndex:f-1
    repString(end+1) = str(j);
  end
  % Insert toStr:
  for j = 1:toStrLength
    repString(end+1) = toStr(j);
  end
  prevIndex = f + fromStrLength;
end
% Insert the last segment:
for j = prevIndex:length(str)
  repString(end+1) = str(j);
end
