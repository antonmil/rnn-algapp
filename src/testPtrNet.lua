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



checkFileExist(sopt.model,'model file')
if lfs.attributes('/home/h3/','mode') then sopt.suppress_x=1 end
print('Loading model ... '..sopt.model)
checkpoint = torch.load(sopt.model)

protosD = checkpoint.protos
protosE = checkpoint.protosE

protosE.rnn:evaluate()
protosD.rnn:evaluate()

opt = checkpoint.opt
------ Change some options for testing
opt.mini_batch_size = 1
opt.gpuid=-1
opt.synth_training, opt.synth_valid = 2,2

opt.max_n=3
opt.max_m=opt.max_n
opt.nClasses = opt.max_m

opt.inSize = opt.max_n * opt.nClasses -- input feature vector size (Linear Assignment)
if opt.problem=='quadratic' then
  opt.inSize = opt.max_n*opt.max_m * opt.max_n*opt.max_m -- QBP
end
opt.rnn_size_encoder = opt.rnn_size

opt.solSize = opt.max_n -- integer
if opt.solution == 'distribution' then
  opt.solSize = opt.max_n*opt.max_m -- one hot (or full)
end

opt.featmat_n, opt.featmat_m = opt.max_n, opt.max_m
if opt.problem == 'quadratic' then 
  opt.featmat_n, opt.featmat_m = opt.max_n * opt.max_m, opt.max_n * opt.max_m
end

opt.TE = opt.featmat_n*(opt.featmat_m+1)


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

-- insert tokens into costs
pm('inserting tokens')
ValCostTabT = insertTokens(ValCostTab)



--if opt.problem == 'linear' then
--  ValProbTab,ValSolTab = genHunData(1)
--elseif opt.problem == 'quadratic' then
--  _,_,ValProbTab,ValSolTab = readQBPData('test')
--end
local nthSample=2

TRAINING=false

costs = ValCostTab[nthSample]:clone()
costsT = ValCostTabT[nthSample]:clone()
huns = ValSolTab[nthSample]:clone()

----- FORWARD ----
-- ENCODE  
local initStateGlobal = clone_list(init_state)
local rnn_stateE = {[0] = initStateGlobal}
local TE = opt.TE


for t=1,TE do
  local rnninp, rnn_stateE = getRNNEInput(t, rnn_stateE)    -- get combined RNN input table
  local lst = protosE.rnn:forward(rnninp) -- do one forward tick
  rnn_stateE[t] = {}
  for i=1,#init_state do table.insert(rnn_stateE[t], lst[i]) end -- extract the state, without output    
end

local rnn_state = {[0] = rnn_stateE[#rnn_stateE]} 
local predictions = {}
local loss = 0
local DA = {}
local T = opt.max_n
local GTDA = {}




for t=1,T do
--   clones.rnn[t]:evaluate()     -- set flag for dropout
  local rnninp, rnn_state = getRNNDInput(t, rnn_state, predictions)    -- get combined RNN input table
--  print(rnninp)
  local lst = protosD.rnn:forward(rnninp)  -- do one forward tick
  predictions[t] = lst
  predictions[t] = {}
  for k,v in pairs(lst) do predictions[t][k] = v:clone() end -- deep copy

  rnn_state[t] = {}
  for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
  print(predictions)
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

