require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'
require 'util.misc'
--require 'external.simple-kalman'

require 'aux'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Test marginals')
cmd:text()
cmd:text('Options')
-- main options
cmd:option('-model_name','trainHun','main model name')
cmd:option('-model_sign','mt1_r100_l1_n4_m4_val','model signature')
cmd:option('-seed',12,'Random seed')
cmd:text()
-- parse input params
sopt = cmd:parse(arg)


torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(sopt.seed)
torch.manualSeed(888)

sopt.model = 'bin/'..sopt.model_name
if sopt.model_sign ~= '' then sopt.model = sopt.model..'_'..sopt.model_sign end
sopt.model = sopt.model..'.t7'



if not lfs.attributes(sopt.model, 'mode') then
    print('Error: File ' .. sopt.model .. ' does not exist.?')
end
if lfs.attributes('/home/h3/','mode') then sopt.suppress_x=1 end
print('Loading model ... '..sopt.model)
checkpoint = torch.load(sopt.model)

protos = checkpoint.protos
protos.rnn:evaluate()
opt = checkpoint.opt
------ Change some options for testing
opt.mini_batch_size = 1

init_state = getInitState(opt, miniBatchSize)
solTable =  findFeasibleSolutions(opt.max_n, opt.max_m)
ValProbsTab,ValHunTab = genHunData(1)




----- FORWARD ----   
local initStateGlobal = clone_list(init_state)  
local rnn_state = {[0] = initStateGlobal}
--   local predictions = {[0] = {[opt.updIndex] = detections[{{},{t}}]}}
local predictions = {}
local loss = 0
local DA = {}
local T = opt.max_n
local GTDA = {}


probs = ValProbsTab[1]:clone()
huns = ValHunTab[1]:clone()
--print(probs)
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



--local costs = torch.rand(3,3)

--print(predDA)
--print(probToCost(predDA))
printDebugValues(probs:reshape(opt.max_n,opt.max_m), costToProb(-predDA))

