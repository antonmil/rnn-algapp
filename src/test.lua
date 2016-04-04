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
sopt.model = sopt.model..'.t7'



if not lfs.attributes(sopt.model, 'mode') then
    print('Error: File ' .. sopt.model .. ' does not exist.?')
    abort('...')
end
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
solTable =  findFeasibleSolutions(opt.max_n, opt.max_m)

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
  ValSolTab = computeMarginals(ValCostTab)
end



--if opt.problem == 'linear' then
--  ValProbTab,ValSolTab = genHunData(1)
--elseif opt.problem == 'quadratic' then
--  _,_,ValProbTab,ValSolTab = readQBPData('test')
--end
local nthSample=2




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

printDebugValues(inpVec, predDA)

