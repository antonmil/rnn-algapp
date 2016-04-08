require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'
require 'util.misc'
require 'mattorch'

require 'aux'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Test marginals')
cmd:text()
cmd:text('Options')
-- main options
cmd:option('-model_name','trainHun','main model name')
cmd:option('-model_sign','mt1_r100_l1_n3_m3_val','model signature')
cmd:option('-seed',12,'Random seed')
cmd:text()
-- parse input params
sopt = cmd:parse(arg)


torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(sopt.seed)

sopt.model = getRootDir()..'bin/'..sopt.model_name
if sopt.model_sign ~= '' then sopt.model = sopt.model..'_'..sopt.model_sign end
sopt.model = sopt.model..'_val.t7'



checkFileExist(sopt.model,'model file')
if lfs.attributes('/home/h3/','mode') then sopt.suppress_x=1 end
print('Loading model ... '..sopt.model)
checkpoint = torch.load(sopt.model)

protos = checkpoint.protos

protos.rnn:evaluate()
opt = checkpoint.opt
------ Change some options for testing
opt.mini_batch_size = 1
opt.gpuid=-1
opt.synth_training, opt.synth_valid = 2,2

init_state = getInitState(opt, miniBatchSize)
solTable = nil

pm('getting training/validation data...')
if opt.problem == 'linear' then
    ValCostTab,ValSolTab = genHunData(opt.synth_valid)
elseif opt.problem == 'quadratic' then
  _,_,ValCostTab,ValSolTab = readQBPData('test')
end


-- normalize to [0,1]
pm('normalizing...')
ValCostTab = normalizeCost(ValCostTab)  

if opt.inference == 'marginal' then
  pm('Computing marginals...')
  solTable =  findFeasibleSolutions(opt.max_n, opt.max_m)
  ValSolTab = computeMarginals(ValCostTab)
end

------ try real data
   local Qfile = string.format('%sdata/test_%d',getRootDir(),opt.max_n);
  checkFileExist(Qfile..'.mat','Q cost file')  
  local loaded = mattorch.load(Qfile..'.mat')
  local allQ = loaded.allQ:t() -- transpose because Matlab is first-dim-major (https://groups.google.com/forum/#!topic/torch7/qDIoWnJzkcU)
  allQ=allQ:float()
  ValCostTab[1]=allQ:reshape(1,opt.inSize)
--  ValCostTab[1]=torch.rand(1,opt.inSize)
  if opt.inference == 'marginal' then 
    ValSolTab = computeMarginals(ValCostTab)
  elseif opt.inference == 'map' then
    ValSolTab[1] =torch.linspace(1,opt.max_n,opt.max_n):reshape(1,opt.max_n)
  end 

--if opt.problem == 'linear' then
--  ValProbTab,ValSolTab = genHunData(1)
--elseif opt.problem == 'quadratic' then
--  _,_,ValProbTab,ValSolTab = readQBPData('test')
--end
local nthSample=1




----- FORWARD ----   
local initStateGlobal = clone_list(init_state)  
local rnn_state = {[0] = initStateGlobal}
--   local predictions = {[0] = {[opt.updIndex] = detections[{{},{t}}]}}
local predictions = {}
local loss = 0
local DA = {}
local T = opt.max_n
local GTDA = {}


costs = ValCostTab[nthSample]:clone()
huns = ValSolTab[nthSample]:clone()
--print(huns:reshape(opt.max_n, opt.max_m))
--abort()
--
--local probMatrix = torch.Tensor({{0.6, 0.35, 0.05},{0.4, 0.21, 0.39},{0.33, 0.32, 0.35}})
--costs = -torch.log(probMatrix)
--probs = torch.exp(-probs):reshape(opt.max_n, opt.max_m)
--print(costs)

--for i=1,opt.max_n do
--print(costs)
--  probs[i] = probs[i] / torch.sum(probs[i])
--end
--print(costs)
--print(torch.sum(costs,2))
--probs = -torch.log(probs)
--print(costs)

for t=1,T do
--   clones.rnn[t]:evaluate()     -- set flag for dropout
  local rnninp, rnn_state = getRNNInput(t, rnn_state, predictions)    -- get combined RNN input table
--  print(rnninp)
  local lst = protos.rnn:forward(rnninp)  -- do one forward tick
  predictions[t] = lst
  predictions[t] = {}
  for k,v in pairs(lst) do predictions[t][k] = v:clone() end -- deep copy

  rnn_state[t] = {}
  for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
  DA[t] = decode(predictions, t)

end

local predDA = decode(predictions):reshape(opt.max_n,opt.nClasses)
predDA=costToProb(-predDA)


--local costs = torch.rand(3,3)
--print(opt.inSize)
--print(predDA)
--print(probToCost(predDA))
local inpVec = costs:clone()
if opt.problem=='linear' then inpVec=inpVec:reshape(opt.max_n,opt.max_m)
elseif opt.problem=='quadratic' then inpVec=inpVec:reshape(opt.max_n*opt.max_m,opt.max_n*opt.max_m)
end

--printDebugValues(inpVec, predDA)

plotProgress(predictions,0,'Test')

