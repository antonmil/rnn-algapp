--require('debugger')(nil,10002);

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'lfs'
require 'image'
require 'mattorch'

torch.setdefaulttensortype('torch.FloatTensor')

local model_utils = require 'util.model'    -- model specific utils
require 'util.misc'   -- all the rest
local RNN = require 'model.RNN'

-- project specific auxiliary functions
require 'aux'

-- for deterministic processing
torch.manualSeed(1)


-- for graph debugging
nngraph.setDebug(true)

-- global timer
ggtime = torch.Timer()


-- option and parameters parsing
cmd = torch.CmdLine()
cmd:text()
cmd:text('Learn an algorithm')
cmd:text()
cmd:text('Options')
-- model params
cmd:option('-config', '', 'config file')
cmd:option('-rnn_size', 10, 'size of RNN internal state')
cmd:option('-model','lstm','module type <RNN|LSTM|GRU>')
cmd:option('-num_layers',1,'number of layers in the RNN / LSTM')
cmd:option('-max_n',2,'number of rows')
cmd:option('-max_m',2,'number of columns')
cmd:option('-lambda',1,'loss weighting')
cmd:option('-problem','quadratic','[linear|quadratic]')
cmd:option('-inference','map','[map|marginal]')
cmd:option('-solution','integer','[integer|distribution]')
cmd:option('-sparse',0,'are the features passed as a sparse matrix?')
cmd:option('-invert_input',0,'Invert input? (Sutskever et al., 2014)')
cmd:option('-double_input',0,'Double input? (Zaremba and Sutskever, 2014)')
cmd:option('-full_input',1,'Input full cost matrix (1), or row-by-row (0)')
cmd:option('-supervised',1,'Supervised or unsupervised learning')
cmd:option('-grad_replace',1,'Replace better predictions with 0 gradient')
cmd:option('-diverse_solutions',1,'take an improved m-bst solution iteratively')
-- optimization
cmd:option('-lrng_rate',1e-2,'learning rate')
cmd:option('-lrng_rate_decay',0.99,'learning rate decay')
cmd:option('-lrng_rate_decay_after',-1,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.97,'decay rate for rmsprop')
cmd:option('-dropout',0.02,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-max_epochs',10000,'number of full passes through the training data')
cmd:option('-grad_clip',.1,'clip gradients at this value')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-training_file', '', 'training data')
cmd:option('-mini_batch_size',1,'mini-batch size')
cmd:option('-rand_par_rng',0.1,'random parameter range')
cmd:option('-forget_bias',1,'forget get biases')
cmd:option('-use_gt_input', 1, 'use gt as current input at t')
cmd:option('-full_set', 1, 'use full (or reduced [0]) training / val set')
cmd:option('-random_epoch', 0, 'random training data each epoch')
-- data related
cmd:option('-synth_training',100,'number of synthetic scenes to augment training data with')
cmd:option('-pert_training',0,'number of perturbed scenes to augment training data with')
cmd:option('-synth_valid',10,'number of synthetic scenes to augment validation data with')
cmd:option('-real_data', 0, 'use (1) or don\'t use (0) real data for validation')
cmd:option('-miss_thr', 0.1, 'threshold for missed detection')
cmd:option('-permute', 0, 'use permutations of the same data?')
cmd:option('-dummy_noise', 0, 'add random noise to const cost')
-- bookkeeping
cmd:option('-seed',122,'torch manual random number generator seed')
cmd:option('-print_every',10,'how many steps/mini-batches between printing out the loss')
cmd:option('-plot_every',10,'how many steps/mini-batches between plotting training')
cmd:option('-eval_val_every',100,'every how many iterations should we evaluate on validation data?')
cmd:option('-save_plots',0,'save plots as png')
cmd:option('-profiler',0,'profiler on/off')
cmd:option('-verbose',2,'Verbosity level')
-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
if string.len(opt.config) > 0 then opt = parseConfig(opt.config, opt) end

checkCuda()       -- check for cuda availability

-- create auxiliary directories (or make sure they exist)
createAuxDirs()


-- augment and fix opt --
opt.supervised=1
opt = fixOpt(opt)
energyOffset = 10
--opt.outSize = opt.nClasses

-- number of hidden inputs (1 for RNN, 2 for LSTM)
opt.nHiddenInputs = 1
if opt.model=='lstm' then opt.nHiddenInputs = 2 end
opt.daPredIndex = opt.num_layers*opt.nHiddenInputs+1
print('da index     '..opt.daPredIndex)

modelName = 'default'   -- base name
if string.len(opt.config)>0 then
  local fp,fn,fe = fileparts(opt.config); modelName = fn
end

modelParams = {'model_index', 'rnn_size', 'num_layers','max_n','max_m',
  'order','sol_index','inf_index'}
dataParams={'synth_training','synth_valid','mini_batch_size',
  'max_n','max_m','state_dim','full_set','fixed_n',
  'temp_win','real_data','real_dets','trim_tracks'}


-- create (or load) prototype
local do_random_init = true
local itOffset = 0
if string.len(opt.init_from) > 0 then
  print('loading an '..opt.model..' from checkpoint ' .. opt.init_from)
  local checkpoint = torch.load(opt.init_from)
  protos = checkpoint.protos

  -- need to adjust some crucial parameters according to checkpoint
  pm('overwriting ...',2)  
  for _, v in pairs(modelParams) do
    if opt[v] ~= checkpoint.opt[v] then
      opt[v] = checkpoint.opt[v]
      pm(string.format('%15s',v) ..'\t = ' .. checkpoint.opt[v], 2)
    end
  end
  pm('            ... based on the checkpoint.',2)
  do_random_init = false
  itOffset = checkpoint.it
  pm('Resuming from iteration '..itOffset+1,2)
else
  print('creating an '..opt.model..' with ' .. opt.num_layers .. ' layers')
  protos = {}
  protos.rnn = RNN.rnn(opt)
  local lambda = opt.lambda
  local nllc = nn.ClassNLLCriterion()
  local bce = nn.BCECriterion()
  local mse = nn.MSECriterion()
  local abserr = nn.AbsCriterion()
  local kl = nn.DistKLDivCriterion()
  local mlm = nn.MultiLabelMarginCriterion()
  local mlsm = nn.MultiLabelSoftMarginCriterion()
  protos.criterion = nn.ParallelCriterion()
  if opt.solution == 'integer' then
    protos.criterion:add(nllc, opt.lambda)
--    protos.criterion:add(bce, opt.lambda)
  elseif opt.solution == 'distribution' then
    protos.criterion:add(kl, opt.lambda)
--    protos.criterion:add(bce, opt.lambda)
  end
end

------- GRAPH -----------
if not onNetwork() then
  print('Drawing Graph')
  graph.dot(protos.rnn.fg, 'Forw', getRootDir()..'graph/RNNForwardGraph_AA')
  graph.dot(protos.rnn.bg, 'Backw', getRootDir()..'graph/RNNBackwardGraph_AA')
end
-------------------------

local _,_,_,modelSign = getCheckptFilename(modelName, opt, modelParams)
local outDir = string.format('%stmp/%s_%s',getRootDir(),modelName, modelSign)
mkdirP(outDir)

-- the initial state of the cell/hidden states
init_state = getInitState(opt, opt.mini_batch_size)

-- ship the model to the GPU if desired
for k,v in pairs(protos) do v = dataToGPU(v) end


-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
if do_random_init then params:uniform(-opt.rand_par_rng, opt.rand_par_rng) end
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' then
  for layer_idx = 1, opt.num_layers do
    for _,node in ipairs(protos.rnn.forwardnodes) do
      if node.data.annotations.name == "i2h_" .. layer_idx then
        print('setting forget gate biases to '..opt.forget_bias..' in LSTM layer ' .. layer_idx)
        -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
        node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(opt.forget_bias)
      end
    end
  end
end
print('number of parameters in the model: ' .. params:nElement())

-- make a bunch of clones after flattening, as that reallocates memory
pm('Cloning '..opt.max_n..' times...',2)
clones = {}
for name,proto in pairs(protos) do
  print('\tcloning ' .. name)
  clones[name] = model_utils.clone_many_times(proto, opt.max_n, not proto.parameters)
end
pm('   ...done',2)

solTable = nil
if opt.inference == 'marginal' then
  solTable =  findFeasibleSolutions(opt.max_n, opt.max_m) -- get feasible solutions
end
  



--local mBst = 1
--local cmdstr = string.format('sh genData.sh %d %d %d %d',opt.max_n, opt.mini_batch_size, opt.synth_training,mBst) 
--print(cmdstr)
--os.execute(cmdstr)
getData(opt, true, true)


--print(TrCostTab[1])
--abort()

--
--print(ValProbTab[1])
--print(ValSolTab[1])

--abort()

function getGTDA(tar)
  local DA = nil


  if opt.solution == 'integer' then
    -- integer
    DA = huns[{{},{tar}}]:reshape(opt.mini_batch_size)
  elseif opt.solution == 'distribution' then
    -- one hot (or full prob distr.)
    local offset = opt.max_n * (tar-1)+1
    DA = huns[{{},{offset, offset+opt.max_n-1}}]
  end
    
--  local offset = opt.max_n * (tar-1)+1
--  DA = huns[{{},{offset, offset+opt.max_n-1}}]  

  return DA
end


function getPredAndGTTables(predDA, tar)
  local input, output = {}, {}

  local GTDA = getGTDA(tar)
  table.insert(input, predDA)
  table.insert(output, GTDA)

  return input, output
end



function eval_val()

  local tL = tabLen(ValCostTab)
  local loss = 0
  local T = opt.max_n
  local plotSeq = math.random(tL)
  plotSeq=1
  local eval_predMeanEnergy = 0
  local eval_predMeanEnergyProj = 0
  for seq=1,tL do
    costs = ValCostTab[seq]:clone()
    huns = ValSolTab[seq]:clone()

    TRAINING = false
    ----- FORWARD ----
    local initStateGlobal = clone_list(init_state)
    local rnn_state = {[0] = initStateGlobal}
    --   local predictions = {[0] = {[opt.updIndex] = detections[{{},{t}}]}}
    local predictions = {}
    --     local loss = 0
    local DA = {}

    local GTDA = {}


    for t=1,T do
      clones.rnn[t]:evaluate()      -- set flag for dropout
      local rnninp, rnn_state = getRNNInput(t, rnn_state, predictions)    -- get combined RNN input table
      local lst = clones.rnn[t]:forward(rnninp) -- do one forward tick
      predictions[t] = lst
      --           print(lst)
      --           abort()

      rnn_state[t] = {}
      for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      DA[t] = decode(predictions, t)
      --     print(DA[t])
      local input, output = getPredAndGTTables(DA[t], t)

      local tloss = clones.criterion[t]:forward(input, output)

      loss = loss + tloss
      

    end
    local predDA = decode(predictions)
    
    local mv, mi = torch.max(predDA,3)
    local sol = getOneHotLab2(mi, true, opt.max_n)
    local predConstr = evalBatchConstraints(sol)
    sol = sol:reshape(opt.mini_batch_size, opt.max_n*opt.nClasses, 1) -- make batches of column vectors
    local cmtr = nil
    if opt.problem=='linear' then
      cmtr = costs:reshape(opt.mini_batch_size, 1, opt.featmat_n*opt.featmat_m)
    elseif opt.problem=='quadratic' then
      cmtr = costs:reshape(opt.mini_batch_size, opt.featmat_n, opt.featmat_m)
    end    
    local predObj = evalBatchEnergy(sol, cmtr)

    local predEnergy = predObj - predConstr * opt.mu
--    print((5-torch.mean(predEnergy,1):squeeze()))
    eval_predMeanEnergy = eval_predMeanEnergy + (torch.mean(predEnergy,1):squeeze())
    
  -- PROJECT FOR PLOTTING
    local projSol = zeroTensor3(opt.mini_batch_size, opt.max_n * opt.max_m,1)
    for m=1,opt.mini_batch_size do
      local ass = hungarianL(-predDA[m]):narrow(2,2,1):t()
      projSol[m] = getOneHotLab(ass,true,opt.max_n):reshape(opt.max_n * opt.max_m,1)
    end  
    local predEnergy = evalBatchEnergy(projSol, cmtr)
    eval_predMeanEnergyProj = eval_predMeanEnergyProj + (torch.mean(predEnergy,1):squeeze())            


    if seq==plotSeq then
      print('Validation checkpoint')
      eval_val_mm = plotProgress(predictions,3,'Validation')
    end

    --         eval_val_mm = 0
    eval_val_multass = 0
  end
  loss = loss / T / tL  -- make average over all frames
  eval_predMeanEnergy = eval_predMeanEnergy / tL
  eval_predMeanEnergyProj = eval_predMeanEnergyProj / tL

  return loss, eval_predMeanEnergy, eval_predMeanEnergyProj
end




seqCnt=0
function feval()
  grad_params:zero()

  local tL = tabLen(TrCostTab)

  seqCnt=seqCnt+1
  if seqCnt > tabLen(TrCostTab) then seqCnt = 1 end
  randSeq = seqCnt

  --  randSeq = math.random(tL) -- pick a random sequence from training set
  costs = TrCostTab[randSeq]:clone()
  huns = TrSolTab[randSeq]:clone()
--  print(huns)
--  abort()

  TRAINING = true
  ----- FORWARD ----
  local initStateGlobal = clone_list(init_state)
  local rnn_state = {[0] = initStateGlobal}
  --   local predictions = {[0] = {[opt.updIndex] = detections[{{},{t}}]}}
  local predictions = {}
  local loss = 0
  local DA = {}
  local T = opt.max_n
  local GTDA = {}
  local rnninp = nil

  for t=1,T do
    clones.rnn[t]:training()      -- set flag for dropout
    rnninp, rnn_state = getRNNInput(t, rnn_state, predictions)    -- get combined RNN input table
    local lst = clones.rnn[t]:forward(rnninp) -- do one forward tick
    predictions[t] = lst
    
    rnn_state[t] = {}
    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
    DA[t] = decode(predictions, t)
--    local input, output = getPredAndGTTables(DA[t], t)
--
--    local tloss = clones.criterion[t]:forward(input, output)
--    loss = loss + tloss
  end
--  loss = loss / T -- make average over all frames
  local predDA = decode(predictions)
  predMeanEnergy, GTMeanEnergy = 0,0
  predMeanEnergyProj, GTMeanEnergyProj = 0,0
    
  local mv, mi = torch.max(predDA,3)
  local sol = getOneHotLab2(mi, true, opt.max_n)

  
  local predConstr = evalBatchConstraints(sol)
--  print(predConstr)
  sol = sol:reshape(opt.mini_batch_size, opt.max_n*opt.nClasses, 1) -- make batches of column vectors
  local cmtr = nil
  if opt.problem=='linear' then
    cmtr = costs:reshape(opt.mini_batch_size, 1, opt.featmat_n*opt.featmat_m)
  elseif opt.problem=='quadratic' then
    cmtr = costs:reshape(opt.mini_batch_size, opt.featmat_n, opt.featmat_m)
  end  
  local predObj = evalBatchEnergy(sol, cmtr)
  
  
  -- get ground truth energy
  local hOneHot = huns:clone()
  if opt.solution == 'distribution' then _,hOneHot = hOneHot:reshape(opt.mini_batch_size, opt.max_n, opt.max_m):max(3) end -- make integer (from max)
--  if opt.solution == 'integer' then hOneHot =  getOneHotLab2(huns, opt.mini_batch_size>1, opt.nClasses) end
  hOneHot =  getOneHotLab2(hOneHot, opt.mini_batch_size>1, opt.nClasses) -- make binary solutions
--  print(hOneHot)
--  abort()
  

  local GTConstr = evalBatchConstraints(hOneHot:reshape(opt.mini_batch_size, opt.max_n, opt.max_m))
  local solGT = hOneHot:reshape(opt.mini_batch_size, opt.max_n*opt.nClasses, 1) -- make batches of column vectors
  local GTObj = evalBatchEnergy(solGT, cmtr)  
  
  local predEnergy = predObj - predConstr * opt.mu
  local GTEnergy = GTObj - GTConstr * opt.mu
  predMeanEnergy = torch.mean(predEnergy,1):squeeze()
  GTMeanEnergy = torch.mean(GTEnergy,1):squeeze()   
  
  local energies = GTEnergy:cat(predEnergy, 3) -- [GT, PRED]
  
  local solv, takeSol = torch.max(energies, 3)
  takeSol=takeSol:reshape(opt.mini_batch_size, 1):eq(2) -- keep prediction?
--  print(torch.sum(takeSol)/opt.mini_batch_size*100 .. '% of GT replaced by prediction')

  -- PROJECT FOR PLOTTING
  local projSol = zeroTensor3(opt.mini_batch_size, opt.max_n * opt.max_m,1)
  local projGT = zeroTensor3(opt.mini_batch_size, opt.max_n * opt.max_m,1)
  for m=1,opt.mini_batch_size do
    local ass = hungarianL(-predDA[m]):narrow(2,2,1):t()
    projSol[m] = getOneHotLab(ass,true,opt.max_n):reshape(opt.max_n * opt.max_m,1)
    if opt.solution == 'distribution' then
      ass = hungarianL(huns[m]:reshape(opt.max_n,opt.max_m)):narrow(2,2,1):t()
    else      
      ass = huns[m]:reshape(1,opt.max_m)
    end
    projGT[m] = getOneHotLab(ass,true,opt.max_n):reshape(opt.max_n * opt.max_m,1)
  end
  local predEnergy = evalBatchEnergy(projSol, cmtr)
  local GTEnergy = evalBatchEnergy(projGT, cmtr)
  
  predMeanEnergyProj = torch.mean(predEnergy,1):squeeze()
  GTMeanEnergyProj = torch.mean(GTEnergy,1):squeeze()
--  print(predMeanEnergyProj)
--  abort()

  if opt.grad_replace ~= 0 then
    if opt.solution=='distribution' then
      if opt.inference == 'map' then 
        huns[takeSol:expand(opt.mini_batch_size, opt.max_n*opt.nClasses)] = sol:clone()
      elseif opt.inference == 'marginal' then
        if opt.project_valid ~= 0 then
          huns[takeSol:expand(opt.mini_batch_size, opt.max_n*opt.nClasses)] = predDA:reshape(opt.mini_batch_size, opt.max_n* opt.nClasses)
        else
          huns[takeSol:expand(opt.mini_batch_size, opt.max_n*opt.nClasses)] = torch.exp(predDA:reshape(opt.mini_batch_size, opt.max_n* opt.nClasses))
        end
      end
    elseif opt.solution == 'integer' then
      huns[takeSol:expand(opt.mini_batch_size, opt.max_n)] = mi:reshape(opt.mini_batch_size, opt.nClasses)
    end    
  end
--  print(huns)


  -- plotting
  if (globiter == 1) or (globiter % opt.plot_every == 0) then
    print('Training checkpoint')
--    print(predictions)
    feval_mm = plotProgress(predictions,1,'Train')
    --    feval_mm = getDAErrorHUN(predDA:reshape(opt.max_n,1,opt.nClasses), hun:reshape(opt.max_n,1))
--    if globiter>1 then abort() end
  end
  
  for t=1,T do
    local input, output = getPredAndGTTables(DA[t], t)--
    local tloss = clones.criterion[t]:forward(input, output)
    loss = loss + tloss
  end
  loss = loss / T -- make average over all frames


  ------ BACKWARD ------
  local rnn_backw = {}
  -- gradient at final frame is zero
  local drnn_state = {[T] = clone_list(init_state, true)} -- true = zero the clones
  for t=T,1,-1 do
    local input, output = getPredAndGTTables(DA[t], t)
    local dl = clones.criterion[t]:backward(input,output)
    local nGrad = #dl

    for dd=1,nGrad do
      table.insert(drnn_state[t], dl[dd]) -- gradient of loss at time t
    end

    local rnninp = getRNNInput(t, rnn_state, predictions)    -- get combined RNN input table

    dlst = clones.rnn[t]:backward(rnninp, drnn_state[t])

    drnn_state[t-1] = {}
    local maxk = opt.num_layers+1
    if opt.model == 'lstm' then maxk = 2*opt.num_layers+1 end
    for k,v in pairs(dlst) do
      if k > 1 and k <= maxk then -- k == 1 is gradient on x, which we dont need
        -- note we do k-1 because first item is dembeddings, and then follow the
        -- derivatives of the state, starting at index 2. I know...
        drnn_state[t-1][k-1] = v
      end
    end
    -- TODO transfer final state?    

  end
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  
  --   feval_mm = 0
  feval_multass = 0
--  print(grad_params[1])

  return loss, grad_params

end



train_energies, val_energies, gt_energies = {}, {}, {}
train_energies_proj, val_energies_proj, gt_energies_proj = {}, {}, {}
train_losses = {}
val_losses = {}
real_losses = {}
train_mm, val_mm, real_mm = {}, {}, {}
train_ma, val_ma, real_ma = {}, {}, {}
mot15_mota = {}
mot15_mota_nopos = {}
local mBstMar = 0
local optim_state = {learningRate = opt.lrng_rate, alpha = opt.decay_rate}
local glTimer = torch.Timer()
for i = 1, opt.max_epochs do
  local epoch = i
  globiter = i
  
--  if i>1 and (i-1)%opt.synth_training==0 and opt.random_epoch~=0 then

  -- replace GT  
  if opt.inference == 'marginal' and opt.problem == 'quadratic' and opt.diverse_solutions ~= 0 then
    if i == 1 or (i-1)%opt.synth_training == 0 then
      mBstMar = mBstMar + 1
      if mBstMar>10 then mBstMar = 1 end
--      if opt.diverse_solutions == 0 then mBstMar = 10 end
      pm('Replacing GT with '.. mBstMar .. '-best Marginals')
      local mBstField = string.format('all_%d_BestMarginals',mBstMar)
      for k,v in pairs(TrSolTab_m_BestMarginals[mBstMar]) do TrSolTab[k] = v end
  --    for k,v in pairs(TrSolTab_m_BestMarginals[mBst]) do TrSolTab[k] = v end
    end
  end

  local timer = torch.Timer()
  local _, loss = optim.rmsprop(feval, params, optim_state)
  local time = timer:time().real

  local train_loss = loss[1] -- the loss is inside a list, pop it
  train_losses[i] = train_loss
  train_energies[i] = predMeanEnergy
  gt_energies[i] = GTMeanEnergy
  train_energies_proj[i] = predMeanEnergyProj
  gt_energies_proj[i] = GTMeanEnergyProj
  

  -- exponential learning rate decay
  if i % (torch.round(opt.max_epochs/10)) == 0 and opt.lrng_rate_decay < 1 then
    --         if epoch >= opt.lrng_rate_decay_after then

    --       print('decreasing learning rate')
    local decay_factor = opt.lrng_rate_decay
    optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
    --         end
  end

  -- print training progress
  if i % opt.print_every == 0 then
    printTrainingStats(i, opt.max_epochs, train_loss, grad_params:norm() / params:norm(),
      time, optim_state.learningRate, glTimer:time().real, predMeanEnergy)
  end



  -- evaluate validation, save chkpt and print/plot loss (not too frequently)
  if (i % opt.eval_val_every) == 0 then
    print(os.date())
    -- plot one benchmark sequence
    --       local real_loss = eval_benchmark()
    local real_loss, eval_benchmark_mm= 1,1
    real_losses[i] = real_loss

    plot_real_loss_x, plot_real_loss = getValLossPlot(real_losses)

    -- evaluate validation set
    val_loss, val_energy, val_energy_proj = eval_val()
    val_losses[i] = math.max(val_loss, 1e-5)
    val_energies[i] = val_energy
    val_energies_proj[i] = val_energy_proj
    

    train_mm[i] = feval_mm+1
    val_mm[i] = eval_val_mm+1
    real_mm[i] = eval_benchmark_mm+1

    train_ma[i] = feval_multass+1
    val_ma[i] = eval_val_multass+1

    local plot_loss_x, plot_loss = getLossPlot(i, opt.eval_val_every, train_losses)
    local plot_val_loss_x, plot_val_loss = getValLossPlot(val_losses)

    local plot_train_mm_x, plot_train_mm = getValLossPlot(train_mm)
    local plot_val_mm_x, plot_val_mm = getValLossPlot(val_mm)
    local plot_real_mm_x, plot_real_mm = getValLossPlot(real_mm)

    local plot_train_ma_x, plot_train_ma = getValLossPlot(train_ma)
    local plot_val_ma_x, plot_val_ma = getValLossPlot(val_ma)
    
    local _, plot_energies = getLossPlot(i, opt.eval_val_every, train_energies)  
    local _, plot_val_energies = getValLossPlot(val_energies)
    local _, plot_gt_energies = getLossPlot(i, opt.eval_val_every, gt_energies)
    local _, plot_energies_proj = getLossPlot(i, opt.eval_val_every, train_energies_proj)  
    local _, plot_val_energies_proj = getValLossPlot(val_energies_proj)

    --       print(train_losses)
    --       print(plot_loss)
    local minTrainLoss, minTrainLossIt = minValAndIt(plot_loss)
    local minValidLoss, minValidLossIt = minValAndIt(plot_val_loss)
    local minRealLoss, minRealLossIt = minValAndIt(plot_real_loss)


    local maxTrainEnergy, maxTrainEnergyIt = maxValAndIt(plot_energies)
    local maxValidEnergy, maxValidEnergyIt = maxValAndIt(plot_val_energies)
    local maxGTEnergy, maxGTEnergyIt = maxValAndIt(plot_gt_energies)
    local maxTrainEnergyProj, maxTrainEnergyItProj = maxValAndIt(plot_energies_proj)
    local maxValidEnergyProj, maxValidEnergyItProj = maxValAndIt(plot_val_energies_proj)


    pm('--------------------------------------------------------')
    pm(string.format('%10s%10s%10s%10s%10s%10s','Obj (max)','Training','Valid','GT','Tr-Prj','Val-Prj'))
    pm(string.format('%10s%10.2f%10.2f%10.2f%10.2f%10.2f','Current',plot_energies[-1],val_energy,plot_gt_energies[-1],plot_energies_proj[-1],plot_val_energies_proj[-1]))
    pm(string.format('%10s%10.2f%10.2f%10.2f%10.2f%10.2f','Best',maxTrainEnergy,  maxValidEnergy,maxGTEnergy,maxTrainEnergyProj,maxValidEnergyProj))
    pm(string.format('%10s%10d%10d%10d%10d%10d','Iter',maxTrainEnergyIt,  maxValidEnergyIt, maxGTEnergyIt,maxTrainEnergyItProj, maxValidEnergyItProj))
    pm('--------------------------------------------------------')
    pm(string.format('%10s%10s%10s%10s','Losses','Training','Valid','Real'))
    pm(string.format('%10s%10.5f%10.5f%10.5f','Current',plot_loss[-1],val_loss,real_loss))
    pm(string.format('%10s%10.5f%10.5f%10.5f','Best',minTrainLoss,  minValidLoss, minRealLoss))
    pm(string.format('%10s%10d%10d%10d','Iter',minTrainLossIt,  minValidLossIt, minRealLossIt))

    -- save lowest val energy if necessary
    if maxValidEnergyIt == i then
      pm('*** New max. validation objective found! ***',2)
      local fn, dir, base, signature, ext = getCheckptFilename(modelName, opt, modelParams)
      local savefile  = dir .. base .. '_' .. signature .. '_valen' .. ext
      saveCheckpoint(savefile, tracks, detections, protos, opt, train_losses, glTimer:time().real, i)
      
      -- save energy as txt
      csvWrite(string.format('%s/en_%.1f.txt',outDir,maxValidEnergy),torch.Tensor(1,1):fill(maxValidEnergy))  -- write cost matrix
      csvWrite(string.format('%s/be.txt',outDir,maxValidEnergy),torch.Tensor(1,1):fill(maxValidEnergy)) -- write cost matrix
          
    end
    

    -- check if we started overfitting
    -- first try generating new data
--    if ((i - minValidLossIt) > 2*opt.eval_val_every) and ((i - minValidLossIt) <= 6*opt.eval_val_every) and opt.random_epoch~=0 then
--      if opt.problem == 'quadratic' then
--        -- call matlab to generate new data
--        local cmdstr = string.format('sh genData.sh %d %d %d',opt.max_n, opt.mini_batch_size, opt.synth_training) 
--        os.execute(cmdstr) 
--      end
--      getData(opt, true, false)
--    end
      
    -- heuristic: abort if no val. loss decrease for 3 last outputs
    if ((i - minValidLossIt) > 6*opt.eval_val_every) then
--      print('Validation loss has stalled. Maybe overfitting. Stop.')
--      break
    end

        
--    if ((i - minValidLossIt) > 2*opt.eval_val_every) then
      -- experiment with iterative GT refinement
--      local mBst = i/opt.eval_val_every + 1
--      mBst = mBst + 1
--      print('Now generate marginals with '..mBst..' mBst approximation')
--      local cmdstr = string.format('sh genData.sh %d %d %d %d',opt.max_n, opt.mini_batch_size, opt.synth_training,mBst) 
--      print(cmdstr)
--      os.execute(cmdstr)
--      getData(opt, true, true)
--    end    
    
    local minTrainLoss, minTrainLossIt = minValAndIt(plot_train_mm)
    local minValidLoss, minValidLossIt = minValAndIt(plot_val_mm)
    local minRealLoss, minRealLossIt = minValAndIt(plot_real_mm)

    pm('--------------------------------------------------------')
    pm(string.format('%10s%10s%10s%10s','MissDA','Training','Valid','Real'))
    pm(string.format('%10s%10.2f%10.2f%10.2f','Current',feval_mm+1,eval_val_mm+1,eval_benchmark_mm+1))
    pm(string.format('%10s%10.2f%10.2f%10.2f','Best',minTrainLoss,  minValidLoss, minRealLoss))
    pm(string.format('%10s%10d%10d%10d','Iter',minTrainLossIt,  minValidLossIt, minRealLossIt))
    pm('--------------------------------------------------------')
    

    

    -- save checkpt
    local savefile = getCheckptFilename(modelName, opt, modelParams)
    saveCheckpoint(savefile, tracks, detections, protos, opt, train_losses, glTimer:time().real, i)


    --       os.execute("th rnnTrackerY.lua -model_name testingY -model_sign r50_l2_n3_m3_d2_b1_v0_li1 -suppress_x 0 -length 5 -seq_name TUD-Crossing")
    -- save lowest val if necessary
    if minValidLossIt == i then
      pm('*** New min. validation loss found! ***',2)
      local fn, dir, base, signature, ext = getCheckptFilename(modelName, opt, modelParams)
      local savefile  = dir .. base .. '_' .. signature .. '_val' .. ext
      saveCheckpoint(savefile, tracks, detections, protos, opt, train_losses, glTimer:time().real, i)
    end

    -- save lowest val if necessary


    -- plot
    local en_norm = 1 --opt.mu
    local lossPlotTab = {}
    
    table.insert(lossPlotTab, {"Trng loss",plot_loss_x,plot_loss, 'with linespoints lt 1'})
    table.insert(lossPlotTab, {"Vald loss",plot_val_loss_x, plot_val_loss, 'with linespoints lt 3'})
    --       table.insert(lossPlotTab, {"Real loss",plot_real_loss_x, plot_real_loss, 'linespoints lt 5'})
    table.insert(lossPlotTab, {"Trng MM",plot_train_mm_x, plot_train_mm+.1, 'with points lt 1'})
    table.insert(lossPlotTab, {"Vald MM",plot_val_mm_x, plot_val_mm-.1, 'with points lt 3'})
    --       table.insert(lossPlotTab, {"Real MM",plot_real_mm_x, plot_real_mm, 'points lt 5'})

    --       table.insert(lossPlotTab, {"Trng MA",plot_train_ma_x, plot_train_ma, 'points pt 1'})
    --       table.insert(lossPlotTab, {"Vald MA",plot_val_ma_x, plot_val_ma, 'points pt 3'})

    table.insert(lossPlotTab, {"Trng O-proj",plot_loss_x,plot_energies_proj/en_norm, 'with linespoints lt 7'})
    table.insert(lossPlotTab, {"Vald O-proj",plot_val_loss_x, plot_val_energies_proj/en_norm, 'with linespoints lt 8'})
    
    --  local minInd = math.min(1,plot_loss:nElement())
    local maxY = math.max(torch.max(plot_loss), torch.max(plot_val_loss), torch.max(plot_real_loss),
      torch.max(plot_train_mm), torch.max(plot_val_mm), torch.max(plot_real_mm))*2
    local minY = math.min(torch.min(plot_loss), torch.min(plot_val_loss), torch.min(plot_real_loss),
      torch.min(plot_train_mm), torch.min(plot_val_mm), torch.min(plot_real_mm))/2
      
    
    local minY, maxY = minMax(plot_train_mm, plot_val_mm, plot_energies_proj/en_norm, plot_val_energies_proj/en_norm,
          plot_loss, plot_val_loss)
    
--    minY = math.max(0.001, minY/2)
--    maxY=maxY*2
--    print(minY, maxY)
--    abort()
    local rangeStr = string.format("set xrange [%d:%d];set yrange [%f:%f]",
      opt.eval_val_every-1, i+1, minY, maxY)
    --       rangeStr = string.format("set yrange [%f:%f]", minY, maxY)
    local rawStr = {}
    table.insert(rawStr, rangeStr)
--    table.insert(rawStr, 'set logscale y')

    local winTitle = string.format('Loss-%06d-%06d',itOffset+1,opt.max_epochs+itOffset)
    plot(lossPlotTab, 2, winTitle, rawStr, 1) -- plot and save (true)    
--    gnuplot.raw('unset logscale') -- for other plots

    --- energies plot
    local enPlotTab = {}
--    print(plot_energies)
    table.insert(enPlotTab, {"Trng Obj",plot_loss_x,plot_energies/en_norm, 'with linespoints lt 4'})
    table.insert(enPlotTab, {"Vald Obj",plot_val_loss_x, plot_val_energies/en_norm, 'with linespoints lt 5'})
    table.insert(enPlotTab, {"GT Obj",plot_loss_x, plot_gt_energies/en_norm, 'with linespoints lt 6'})
    
    local minY, maxY = minMax(plot_energies/en_norm, plot_val_energies/en_norm, plot_gt_energies/en_norm)        
--    print(minY, maxY)
    local winTitle = string.format('Objective-%06d-%06d',itOffset+1,opt.max_epochs+itOffset)
    rangeStr = string.format("set xrange [%d:%d];set yrange [%f:%f]", opt.eval_val_every-1, i+1, minY, maxY)
    local rawStr = {}
    table.insert(rawStr, rangeStr)    
    plot(enPlotTab, 5, winTitle, rawStr, 1) -- plot and save (true)


    printModelOptions(opt, modelParams) -- print parameters


  end

  if (i == 1 or i % (opt.print_every*10) == 0) and i<opt.max_epochs then
    printModelOptions(opt, modelParams)
    printTrainingHeadline()       
  end -- headline


  

end

print('-------------   PROFILING   INFO   ----------------')
local totalH, totalM = secToHM(ggtime:time().real)
print(string.format('%20s%5d:%02d%7s','total time',totalH, totalM,''))