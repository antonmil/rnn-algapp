require 'lfs'
require 'util.misc'
require 'gnuplot'
-- project specific auxiliary functions
require 'aux'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Find out best run')
cmd:text()
cmd:text('Options')
cmd:option('-main_name','1013A','main model name')
cmd:option('-subexper','a','subexperiment')
-- cmd:option('-model_sign','r76_l3_n4_d4_b2_lt1','model signature')
cmd:text()
-- parse input params
opt = cmd:parse(arg)
opt.max_m = opt.max_n


-- Code by David Kastrup
function dirtree(dir)
  assert(dir and dir ~= "", "directory parameter is missing or empty")
  if string.sub(dir, -1) == "/" then
    dir=string.sub(dir, 1, -2)
  end

  local function yieldtree(dir)
    for entry in lfs.dir(dir) do
      if entry ~= "." and entry ~= ".." then
        entry=dir.."/"..entry
	local attr=lfs.attributes(entry)
	coroutine.yield(entry,attr)
	if attr.mode == "directory" then
	  yieldtree(entry)
	end
      end
    end
  end

  return coroutine.wrap(function() yieldtree(dir) end)
end


-- trim string
function strim(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

-- replace _ with \-
function replUndie(s)
  return (s:gsub("_","\\_"))
end

-- print(opt.model_name)
-- print(opt.subexper)

-- a list of model-specific parameters... These cannot change across models
modelParams = {'model_index', 'rnn_size', 'num_layers','max_n','max_m',
  'order','sol_index','inf_index'}
dataParams={'synth_training','synth_valid','mini_batch_size',
  'max_n','max_m','state_dim','full_set','fixed_n',
  'temp_win','real_data','real_dets','trim_tracks'}


if (string.find(opt.main_name,'HUN')) ~= nil then 
modelParams = {'model_index','rnn_size', 'num_layers','max_n','max_m'}
end
print(modelParams)

local rangeFile = '../config/'..opt.main_name..'-range.txt'
-- print(rangeFile)
local parRanges = csvRead(rangeFile, nil, true)
for k,v in pairs(parRanges) do
  local subexper = v[1]
  local pname = strim(v[2])
  local from = v[3]
  local to = v[4]
  local n = v[5]
  local logsp = v[6]
  
  
  local allMota = {}
  local allExper = {}
  
  local prange = torch.linspace(from,to,n)
  if logsp ~= 0 then
	prange = torch.exp(torch.linspace(torch.log(from), torch.log(to), n))
  end
  local pstep = prange[2] - prange[1]
  
  for exper=1,n do
    local modelName = opt.main_name..subexper..'-'..exper
    local confFile = '../config/'..modelName..'.txt'
--     print(confFile)
--     abort()
    
    opt = parseConfig(confFile, opt)
    opt.model = 'lstm'
    opt = fixOpt(opt)
--    opt.max_m = opt.max_n 

--     print(opt)
    local _,_,_,modelSign = getCheckptFilename(modelName, opt, modelParams)
    
    local outDir = string.format('../tmp/%s_%s/',modelName, modelSign)
--    print(outDir)
    local bmfile = outDir..'/be.txt'
--     print(bmfile)
    
    if lfs.attributes(bmfile,'mode') then
      local mota = torch.Tensor(csvRead(bmfile)):squeeze() 	-- read result
      table.insert(allMota, mota)
      table.insert(allExper, prange[exper])
    end
--     abort()
  end
  
--   local tmpDir = 
  print(subexper, pname, from, to, n)
  
--   sleep(1)
  
--   print(allMota)
--   print(allExper)
--   abort()
  
  if #allMota>0 and #allExper>0 then

    --       print(#allMota)
--   print(allExper)
    local amTen = torch.Tensor(allMota)
    local aeTen = torch.Tensor(allExper)
    
    local mainconf = '../config/'..opt.main_name..'.txt'  
    opt = parseConfig(mainconf, opt)
    local mainpar = opt[pname] -- default parameter
    local mpX = torch.Tensor({mainpar, mainpar})
    local mpY = torch.Tensor({torch.min(amTen)-1, torch.max(amTen)+1})
    
    local plotTab = {}
    
    local legend='('..subexper..') '..replUndie(pname)
    table.insert(plotTab, {legend, aeTen, amTen, 'points pt 7'})
    table.insert(plotTab, {'default', mpX, mpY,'lines lt 1 lc 2'})
--     print(plotTab)
--     abort()
    -- print(aeTen:view(1,-1))
    -- print(amTen:view(1,-1))
    local srt,srtInd = torch.sort(amTen)
    -- print(srt,srtInd)
    -- local comb = aeTen:view(1,-1):cat(amTen:view(1,-1),1)
    -- print(srt:view(1,-1))
    -- print(aeTen:index(1,srtInd))
    local comb = srt:view(1,-1):cat(aeTen:index(1,srtInd):view(1,-1),1)
    print(comb)
    -- print(plotTab)
    gnuplot.raw(string.format('set term wxt 8',winID))
    gnuplot.raw(string.format('set xrange [%f:%f]',from-pstep, to+pstep))
--     gnuplot.raw(string.format("set autoscale xy"))

    -- if x available, display  
    if xAvailable() then
      gnuplot.plot(plotTab)
      gnuplot.plotflush()
    end

    -- save as png
    gnuplot.raw('set term pngcairo enhanced font "arial,10" fontscale 1.0 size 500,200;')
  --   print(opt.main_name..subexper)
    gnuplot.raw(string.format("set output \"../tmp/%s.png\"",opt.main_name..subexper))
  --   print(plotTab)
    gnuplot.plot(plotTab)
    gnuplot.plotflush()  
  
  end  
--   break
end
-- print(parRanges)
-- a,b=string.find(opt.model_name,"-(%d+)")
  -- if a==nil then a,b=string.find(v,"-(%d+)DA_") end
  
--   local exper = (string.sub(v,a+1,b))
-- abort()


-- which directories exist?
-- local tmpDir = 'tmp/';
-- local modelDirs = {}
-- for filename, attr in dirtree(tmpDir) do
-- --   print(attr.mode, filename)
--   if attr.mode == 'directory' then
--     local substr = string.sub(filename, string.len(tmpDir)+1, string.len(tmpDir)+string.len(opt.model_name))
--     if substr == opt.model_name then
--       table.insert(modelDirs,filename)
--     end
--   end  
-- end

-- collect info
-- local allMota = {}
-- local allExper = {}

-- print(modelDirs)
-- if #modelDirs==0 then return end
-- 
-- for k,v in pairs(modelDirs) do
-- --   print(k,v)
--   local pre = string.format('%s%s-',tmpDir, opt.model_name)
--   a,b=string.find(v,"-(%d+)")
--   -- if a==nil then a,b=string.find(v,"-(%d+)DA_") end
--   
--   local exper = (string.sub(v,a+1,b))
--   -- print(k,v,a,b,exper)
--   local bmfile = v..'/bm.txt'
--   if lfs.attributes(bmfile,'mode') then
--     local mota = torch.Tensor(csvRead(bmfile)):squeeze() 	-- read result
--     table.insert(allMota, mota)
--     table.insert(allExper, tonumber(exper))
--   end
--   
-- end

-- plot


-- local cmd = string.format('ls -l tmp/%s*',opt.model_name)
-- -- print(cmd)
-- local modelDirs, a = os.execute(cmd)
-- print('m',modelDirs, a)
