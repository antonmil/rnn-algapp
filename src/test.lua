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
cmd:option('-suppress_x',0,'suppress plotting in terminal')
cmd:option('-test_file','test','test file name')
cmd:option('-seed',1,'Random seed')
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

protos = checkpoint.protos

protos.rnn:evaluate()
opt = checkpoint.opt
------ Change some options for testing
opt.mini_batch_size = 1
opt.gpuid=-1
opt.synth_training, opt.synth_valid = 1,1
opt.suppress_x = sopt.suppress_x

init_state = getInitState(opt, miniBatchSize)
solTable = nil

pm('getting training/validation data...')
if opt.problem == 'linear' then
    ValCostTab,ValSolTab = genHunData(opt.synth_valid)
elseif opt.problem == 'quadratic' then
  local testfile = string.format('%sdata/%s_%d.mat',getRootDir(),sopt.test_file,opt.max_n);
--  print(testfile)
--  abort()
  _,_,ValCostTab,ValSolTab = readQBPData('test', testfile)
------ try real data
--if opt.problem == 'quadratic' then
--  local Qfile = string.format('%sdata/%s_%d.mat',getRootDir(),sopt.test_file,opt.max_n);
--  checkFileExist(Qfile,'Q cost file')  
--  local loaded = mattorch.load(Qfile)
--  print('test file loaded... '..Qfile)
--
--  local allQ = loaded.allQ:t() -- transpose because Matlab is first-dim-major (https://groups.google.com/forum/#!topic/torch7/qDIoWnJzkcU)
--  allQ=allQ:float()
--  local inSize = opt.inSize; if opt.double_input ~= 0 then inSize=inSize/2 end
--  ValCostTab[1]=allQ:reshape(1,inSize)
----  ValCostTab[1]=torch.rand(1,opt.inSize)
--  if opt.inference == 'marginal' then 
----    ValSolTab = computeMarginals(ValCostTab)
----    print(loaded.allMarginals:t())
----    abort()
--    ValSolTab[1] = loaded.allMarginals:t()
--  elseif opt.inference == 'map' then
--    if opt.solution == 'integer' then
----    ValSolTab[1] =torch.linspace(1,opt.max_n,opt.max_n):reshape(1,opt.max_n)
----print(loaded)
--      ValSolTab[1] = loaded.allSolInt:t()
--    elseif opt.solution == 'distribution' then
--      ValSolTab[1] = loaded.allSol:t()
--    end
--  end 
--end

end


--if opt.inference == 'marginal' then
--  pm('Computing marginals...')
--  solTable =  findFeasibleSolutions(opt.max_n, opt.max_m)
--  ValSolTab = computeMarginals(ValCostTab)
--end


--print(ValCostTab[1]:view(opt.max_n,opt.max_m))
ValCostTab = prepData(ValCostTab)
--print(ValCostTab[1])
--print(ValSolTab[1])
--abort()

--if opt.problem == 'linear' then
--  ValProbTab,ValSolTab = genHunData(1)
--elseif opt.problem == 'quadratic' then
--  _,_,ValProbTab,ValSolTab = readQBPData('test')
--end

local nthSample = sopt.seed
--nthSample = math.random(opt.synth_valid)
--print(math.random(opt.synth_valid))
--print(math.random(opt.synth_valid))
--print(math.random(opt.synth_valid))
--print(opt.synth_valid)
--print(nthSample)


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

local timer = torch.Timer()
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
if opt.supervised == 0 then 
  predDA = predictions[1][opt.daPredIndex+1]:reshape(opt.mini_batch_size*opt.max_n,opt.nClasses):sub(1,opt.max_n)
else
  local logpredDA = decode(predictions):reshape(opt.mini_batch_size*opt.max_n,opt.nClasses):sub(1,opt.max_n)  
  predDA=costToProb(-logpredDA)    
end

local runtime = timer:time().real
pm('Inference time: '..runtime..' seconds.')

--local predDA = decode(predictions):reshape(opt.max_n,opt.nClasses)
--predDA=costToProb(-predDA)
--print(predDA)

local pmaxv,pmaxi = torch.max(predDA,2)
local oneHotPred = getOneHotLab(pmaxi, true, opt.max_m)
--print(oneHotPred)
--local costs = torch.rand(3,3)
--print(opt.inSize)
--print(predDA)
--print(probToCost(predDA))
--local inpVec = costs:clone()
--if opt.problem=='linear' then inpVec=inpVec:reshape(opt.max_n,opt.max_m)
--elseif opt.problem=='quadratic' then inpVec=inpVec:reshape(opt.max_n*opt.max_m,opt.max_n*opt.max_m)
--end

--printDebugValues(inpVec, predDA)

--print(predictions)
plotProgress(predictions,0,'Test')

-- write results into txt
local writeResTensor = oneHotPred:reshape(opt.max_n*opt.max_m,1):cat(predDA:reshape(opt.max_n*opt.max_m,1),2)
local timeVec = torch.zeros(opt.max_n*opt.max_m, 1) timeVec[1][1] = runtime
writeResTensor=writeResTensor:cat(timeVec, 2)
local outFile = getRootDir()..'out'..'/'..sopt.model_name..'_'..sopt.model_sign..'.txt'


csvWrite(outFile,writeResTensor)
pm('results written to '..outFile)
