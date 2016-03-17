require 'torch'
require 'nn'
require 'nngraph'

--------------------------------------------------------------------------
--- each layer has the same hidden input size
function getInitState(opt)
  local init_state = {}
  for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.mini_batch_size, opt.rnn_size)
    table.insert(init_state, dataToGPU(h_init:clone()))
    if opt.model == 'lstm' then
      table.insert(init_state, dataToGPU(h_init:clone()))
    end
  end
  return init_state
end

function getOneCost(seed)
  
  if seed ~= nil then torch.manualSeed(seed) end
  
  local oneCost = torch.rand(opt.mini_batch_size*opt.max_n, opt.nClasses)
--   local fillMatrix = torch.ones(opt.mini_batch_size*opt.max_n,opt.max_m) * opt.miss_thr
--   if opt.dummy_noise ~= 0 then
--     fillMatrix = fillMatrix + torch.rand(opt.mini_batch_size*opt.max_n,opt.max_m) * opt.dummy_noise
--   end
  
  oneCost[{{},{opt.max_m+1,opt.nClasses}}] = getFillMatrix(true, opt.miss_thr, opt.dummy_noise, 1)
  oneCost=oneCost:reshape(opt.mini_batch_size, opt.max_n*opt.nClasses)
  
--   torch.manualSeed(math.random(1e10))
  
  return oneCost
end

function permuteCost(cost)
  
  cost = cost:reshape(opt.mini_batch_size*opt.max_n, opt.nClasses)
  
--   for r=1,cost:size(1) do
  for mb=1,opt.mini_batch_size do
    local per = torch.randperm(opt.max_m)
    local mbStartT = opt.max_n * (mb-1)+1
    local mbEndT =   opt.max_n * mb
  

--     print(per)
    for r=1,opt.max_n do
--       print(cost[{{mbStartT+r-1},{1,opt.max_m}}])
      cost[{{mbStartT+r-1},{1,opt.max_m}}] = cost[{{mbStartT+r-1},{1,opt.max_m}}]:index(2,per:long())
    end
  end
  
  return cost:reshape(opt.mini_batch_size, opt.max_n*opt.nClasses)
  
end

function genHunData(nSamples)
--   if opt.permute ~= 0 then error('permutation not implemented') end
  local CostTab, HunTab = {},{}
  local missThr = opt.miss_thr
  for n=1,nSamples do
    
    if n%math.floor(nSamples/5)==0 then 
      print((n)*(100/(nSamples))..' %...') 
    end
    
    if n==1 or opt.permute<=1 or (opt.permute>1 and n%opt.permute == 0) then
      oneCost = getOneCost()
    else
--       print('permuting?')
      oneCost = permuteCost(oneCost)
    end
    
    
    --     oneCost = oneCost:cat(torch.ones(opt.mini_batch_size, opt.max_n)*missThr)
--     print(oneCost)
--     abort()
    oneCost = dataToGPU(oneCost)
    table.insert(CostTab, oneCost)
    
    local hunSols = torch.ones(1,opt.max_n):int()
    for m=1,opt.mini_batch_size do
      local costmat = oneCost[m]:reshape(opt.max_n, opt.nClasses)    
      local ass = hungarianL(costmat)
      hunSols = hunSols:cat(ass[{{},{2}}]:reshape(1,opt.max_n):int(), 1)    
    end
    hunSols=hunSols:sub(2,-1)
    table.insert(HunTab, hunSols)    
  end
  return CostTab, HunTab
end

--------------------------------------------------------------------------
--- get all inputs for one time step
-- @param t   time step
-- @param rnn_state hidden state of RNN (use t-1)
-- @param predictions current predictions to use for feed back
function getRNNInput(t, rnn_state, predictions)
  local rnninp = {}
  
  -- Cost matrix
  local loccost = costs:clone():reshape(opt.mini_batch_size, opt.max_n*opt.nClasses)
  table.insert(rnninp, loccost)
  
  for i = 1,#rnn_state[t-1] do table.insert(rnninp,rnn_state[t-1][i]) end
  
  return rnninp, rnn_state
end

--------------------------------------------------------------------------
--- RNN decoder
-- @param predictions current predictions to use for feed back
-- @param t   time step (nil to predict for entire sequence)
function decode(predictions, tar)
  local DA = {}
  local T = tabLen(predictions) -- how many targets
  if tar ~= nil then
    local lst = predictions[tar]
    DA = lst[opt.daPredIndex]:reshape(opt.mini_batch_size, opt.nClasses) -- opt.mini_batch_size*opt.max_n x opt.max_m
  else
    DA = zeroTensor3(opt.mini_batch_size,T,opt.nClasses)
    for tt=1,T do
      local lst = predictions[tt]
      DA[{{},{tt},{}}] = lst[opt.daPredIndex]:reshape(opt.mini_batch_size, 1, opt.nClasses) 
    end
  end
  return DA
end