--[[
-- Example use:
file csv1.txt:

1.23,70,hello
there,9.81,102
8,1.243,test

Save the following as test.lua to test this module:
-------------------------------------
local csvfile = require "simplecsv"
local m = csvfile.read('./csv1.txt') -- read file csv1.txt to matrix m
print(m[2][3])                       -- display element in row 2 column 3 (102)
m[1][3] = 'changed'                  -- change element in row 1 column 3
m[2][3] = 123.45                     -- change element in row 2 column 3
csvfile.write('./csv2.txt', m)       -- write matrix to file csv2.txt
-------------------------------------
]]

-- module(..., package.seeall) 
-- author...
-- modified by Anton Milan

---------------------------------------------------------------------
--- An auxiliary function to split strings for CSV files
-- @param str	the input string
-- @param sep	the separator (default ',')
-- @author David Amnon
local function csvSplit(str, sep)
    sep = sep or ','
    fields={}
    local matchfunc = string.gmatch(str, "([^"..sep.."]+)")
    if not matchfunc then return {str} end
    for str in matchfunc do
        table.insert(fields, str)
    end
    return fields
end

---------------------------------------------------------------------
function csvRead(path, sep, tonum)
    tonum = tonum or true
    sep = sep or ','
    local csvFile = {}
    local file = assert(io.open(path, "r"))
    for line in file:lines() do
        fields = csvSplit(line, sep)
        if tonum then -- convert numeric fields to numbers
            for i=1,#fields do
                fields[i] = tonumber(fields[i]) or fields[i]
            end
        end
        table.insert(csvFile, fields)
    end
    file:close()
    return csvFile
end

---------------------------------------------------------------------
function csvWrite(path, data, sep)
    sep = sep or ','
    local file = assert(io.open(path, "w"))
    
    if type(data)=='table' then
      for i=1,#data do
	  for j=1,#data[i] do
	      if j>1 then file:write(sep) end
	      file:write(data[i][j])
	  end
	  file:write('\n')
      end
    elseif torch.isTensor(data) then
      if data:nDimension() < 2 then -- zero solution case
-- 	file:write('\n')
      else
	for i=1,data:size(1) do
	    for j=1,data:size(2) do
		if j>1 then file:write(sep) end
		file:write(data[i][j])
	    end
	    file:write('\n')
	end      
      end
    else
      error('unknown data type in csvwrite')
    end
  
    file:close()
end