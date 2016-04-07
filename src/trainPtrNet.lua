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
local RNNE = require 'model.RNNEncoder'
local RNND = require 'model.RNNDecoder'

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
opt = fixOpt(opt)

-- seq-to-seq specific
opt.nClasses=1
opt.rnn_size_encoder = opt.rnn_size
opt.featmat_n, opt.featmat_m = opt.max_n, opt.max_m
if opt.problem == 'quadratic' then 
  opt.featmat_n, opt.featmat_m = opt.max_n * opt.max_m, opt.max_n * opt.max_m
end

opt.TE = opt.featmat_n*(opt.featmat_m)


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

modelParams = {'model_index', 'rnn_size', 'num_layers','max_n','max_m'}
dataParams={'synth_training','synth_valid','mini_batch_size',
  'max_n','max_m','state_dim','full_set','fixed_n',
  'temp_win','real_data','real_dets','trim_tracks'}

-- create prototype for Encoder
print('creating an encoder '..opt.model..' with ' .. opt.num_layers .. ' layers')
protosE = {}
protosE.rnn = RNNE.rnn(opt)

-- create prototype for Decoder
print('creating a decoder '..opt.model..' with ' .. opt.num_layers .. ' layers')
protosD = {}
protosD.rnn = RNND.rnn(opt)
local nllc = nn.ClassNLLCriterion()
local kl = nn.DistKLDivCriterion()
local mse = nn.MSECriterion()
protosD.criterion = nn.ParallelCriterion()
if opt.solution == 'integer' then
--  protosD.criterion:add(nllc, opt.lambda)
  protosD.criterion:add(mse, opt.lambda)
elseif opt.solution == 'distribution' then
--  protosD.criterion:add(kl, opt.lambda)
  protosD.criterion:add(mse, opt.lambda)
end


------- GRAPH -----------
if not onNetwork() then
  print('Drawing Graph')
  graph.dot(protosE.rnn.fg, 'Forw', getRootDir()..'graph/RNNEForwardGraph_AA')
  graph.dot(protosE.rnn.bg, 'Backw', getRootDir()..'graph/RNNEBackwardGraph_AA')
  graph.dot(protosD.rnn.fg, 'Forw', getRootDir()..'graph/RNNDForwardGraph_AA')
  graph.dot(protosD.rnn.bg, 'Backw', getRootDir()..'graph/RNNDBackwardGraph_AA')
end

-------------------------

local itOffset = 0
-- the initial state of the cell/hidden states
init_state = getInitState(opt, opt.mini_batch_size)

-- ship the model to the GPU if desired
for k,v in pairs(protosE) do v = dataToGPU(v) end
for k,v in pairs(protosD) do v = dataToGPU(v) end


-- put the above things into one flattened parameters tensor
--paramsE, grad_paramsE = model_utils.combine_all_parameters(protosE.rnn)
params, grad_params = model_utils.combine_all_parameters(protosE.rnn,protosD.rnn)
--print(params:size())
--abort()
local do_random_init = true
if do_random_init then 
--  paramsE:uniform(-opt.rand_par_rng, opt.rand_par_rng) 
  params:uniform(-opt.rand_par_rng, opt.rand_par_rng)
end
--params=paramsE:cat(paramsD)
--grad_params=grad_paramsE:cat(grad_paramsD)
--params = paramsD:clone()
--grad_params=grad_paramsD:clone()

-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' then
  for layer_idx = 1, opt.num_layers do
    for _,node in ipairs(protosE.rnn.forwardnodes) do
      if node.data.annotations.name == "i2h_" .. layer_idx then
        print('setting forget gate biases to '..opt.forget_bias..' in LSTM layer ' .. layer_idx)
        -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
        node.data.module.bias[{{opt.rnn_size_encoder+1, 2*opt.rnn_size_encoder}}]:fill(opt.forget_bias)
      end
    end
  end
end
if opt.model == 'lstm' then
  for layer_idx = 1, opt.num_layers do
    for _,node in ipairs(protosD.rnn.forwardnodes) do
      if node.data.annotations.name == "i2h_" .. layer_idx then
        print('setting forget gate biases to '..opt.forget_bias..' in LSTM layer ' .. layer_idx)
        -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
        node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(opt.forget_bias)
      end
    end
  end
end


--print('number of parameters in the encoder model: ' .. paramsE:nElement())
--print('number of parameters in the decoder model: ' .. paramsD:nElement())
print('number of parameters in the entire model : ' .. params:nElement())

-- make a bunch of clones after flattening, as that reallocates memory
pm('Cloning '..opt.max_n..' times...',2)
clonesE, clonesD = {}, {}
for name,proto in pairs(protosE) do
  clonesE[name] = model_utils.clone_many_times(proto, opt.TE, not proto.parameters)
end
clonesD = {}
for name,proto in pairs(protosD) do
  clonesD[name] = model_utils.clone_many_times(proto, opt.TE, not proto.parameters)
end
pm('   ...done',2)

solTable = nil

pm('getting training/validation data...')
if opt.problem == 'linear' then
  ----- gen data for Hungarian
  --  if opt.inference == 'map' then
  TrCostTab,TrSolTab = genHunData(opt.synth_training)
  ValCostTab,ValSolTab = genHunData(opt.synth_valid)
  
  --  elseif opt.inference == 'marginal' then
  --    TrCostTab,TrSolTab = genMarginalsData(opt.synth_training)
  --    ValCostTab,ValSolTab = genMarginalsData(opt.synth_valid)
  --  end
elseif opt.problem == 'quadratic' then
  TrCostTab,TrSolTab,ValCostTab,ValSolTab = readQBPData('train')
end


-- normalize to [0,1]
pm('normalizing...')
TrCostTab = normalizeCost(TrCostTab)
ValCostTab = normalizeCost(ValCostTab)


if opt.inference == 'marginal' then
  pm('Computing marginals...')
  solTable =  findFeasibleSolutions(opt.max_n, opt.max_m) -- get feasible solutions
  TrSolTab = computeMarginals(TrCostTab)
  ValSolTab = computeMarginals(ValCostTab)
end

-- insert tokens into costs
pm('inserting tokens')
TrCostTabT = insertTokens(TrCostTab)
ValCostTabT = insertTokens(ValCostTab)

--if opt.solution == 'integer' then
--  -- convert to one hot
--  TrSolTab = oneHotAll(TrSolTab)
--  ValSolTab = oneHotAll(ValSolTab)
--end

-- set all costs to 1
--opt.inSize = 1 -- WARNING
--for k,v in pairs(TrCostTab) do
--  TrCostTab[k] = dataToGPU(torch.zeros(opt.mini_batch_size, opt.inSize))
--end
--for k,v in pairs(ValCostTab) do
--  ValCostTab[k] = dataToGPU(torch.zeros(opt.mini_batch_size, opt.inSize))
--end


function getGTDA(tar)
  local DA = nil


--  if opt.solution == 'integer' then
--    -- integer
--    DA = huns[{{},{tar}}]:reshape(opt.mini_batch_size)
----    print(DA)
----    abort()
--  elseif opt.solution == 'distribution' then
--    -- one hot (or full prob distr.)    
--    local offset = opt.max_n * (tar-1)+1
--    DA = huns[{{},{offset, offset+opt.max_n-1}}]
--  end
  
--  print(huns)
--  print(tar)
  DA = huns[{{},{tar}}]:reshape(opt.mini_batch_size)

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
  local T = opt.TE
  local plotSeq = math.random(tL)
  plotSeq=1
  for seq=1,tL do
    costs = ValCostTab[seq]:clone()
    costsT = ValCostTabT[seq]:clone()
    huns = ValSolTab[seq]:clone()

    TRAINING = false
    ----- FORWARD ----
    -- ENCODE  
    local initStateGlobal = clone_list(init_state)
    local rnn_stateE = {[0] = initStateGlobal}
    local TE = opt.TE
    
    
    
    for t=1,TE do
      clonesE.rnn[t]:training()
      local rnninp, rnn_stateE = getRNNEInput(t, rnn_stateE)    -- get combined RNN input table
      local lst = clonesE.rnn[t]:forward(rnninp) -- do one forward tick
      rnn_stateE[t] = {}
      for i=1,#init_state do table.insert(rnn_stateE[t], lst[i]) end -- extract the state, without output    
    end


  
    local initStateGlobal = clone_list(init_state)
    local rnn_state = {[0] = rnn_stateE[#rnn_stateE]}
    local predictions = {}
    local DA = {}

    local GTDA = {}


    for t=1,T do
      clonesD.rnn[t]:evaluate()      -- set flag for dropout
      local rnninp, rnn_state = getRNNDInput(t, rnn_state, predictions, rnn_stateE)    -- get combined RNN input table
      local lst = clonesD.rnn[t]:forward(rnninp) -- do one forward tick
      predictions[t] = lst
      --           print(lst)
      --           abort()

      rnn_state[t] = {}
      for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      DA[t] = decode(predictions, t)
      --     print(DA[t])
      local input, output = getPredAndGTTables(DA[t], t)

      local tloss = clonesD.criterion[t]:forward(input, output)

      loss = loss + tloss
    end


    if seq==plotSeq then
      print('Validation checkpoint')
      eval_val_mm = plotProgressD(predictions,3,'Validation')
    end

    --         eval_val_mm = 0
    eval_val_multass = 0
  end
  loss = loss / T / tL  -- make average over all frames

  return loss
end




seqCnt=0
function feval()
--  grad_paramsE:zero()
--  print(grad_paramsE[1])
--  grad_paramsD:zero()
  grad_params:zero()

  local tL = tabLen(TrCostTab)

  seqCnt=seqCnt+1
  if seqCnt > tabLen(TrCostTab) then seqCnt = 1 end
  randSeq = seqCnt

  --  randSeq = math.random(tL) -- pick a random sequence from training set
  costs = TrCostTab[randSeq]:clone()
  costsT = TrCostTabT[randSeq]:clone()
  huns = TrSolTab[randSeq]:clone()

  TRAINING = true
  local rnninp = nil
  local initStateGlobal = clone_list(init_state)
  ----- FORWARD ----
  -- ENCODE    
  local rnn_stateE = {[0] = initStateGlobal}
  local TE = opt.TE
  
  
  for t=1,TE do
    clonesE.rnn[t]:training()
    rnninp, rnn_stateE = getRNNEInput(t, rnn_stateE)    -- get combined RNN input table
    local lst = clonesE.rnn[t]:forward(rnninp) -- do one forward tick
    rnn_stateE[t] = {}
    for i=1,#init_state do table.insert(rnn_stateE[t], lst[i]) end -- extract the state, without output    
  end


--    print(rnn_stateE)
--    abort()
    
  -- DECODE
  local predictions = {}
  local loss = 0
  local DA = {}
--  local T = opt.max_n
  local T = opt.TE
  local GTDA = {}

  local rnn_state = {[0] = rnn_stateE[#rnn_stateE]}
--  local rnn_state = {[0] = initStateGlobal}
--  print(rnn_state[0][1][1][1])
  
  for t=1,T do
    clonesD.rnn[t]:training()      -- set flag for dropout
    rnninp, rnn_state = getRNNDInput(t, rnn_state, predictions, rnn_stateE)    -- get combined RNN input table
    local lst = clonesD.rnn[t]:forward(rnninp) -- do one forward tick
    predictions[t] = lst
    --         print(lst)
    --         abort()

    --    print(rnninp[1])
    rnn_state[t] = {}
    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
    DA[t] = decode(predictions, t)
    local input, output = getPredAndGTTables(DA[t], t)

    local tloss = clonesD.criterion[t]:forward(input, output)
    loss = loss + tloss
  end
  loss = loss / T -- make average over all frames
  local predDA = decode(predictions)

  if (globiter == 1) or (globiter % opt.plot_every == 0) then
    print('Training checkpoint')
    feval_mm = plotProgressD(predictions,1,'Train')
  end


  ------ BACKWARD ------
  -- gradient at final frame is zero
  local drnn_state = {[T] = clone_list(init_state, true)} -- true = zero the clones
  for t=T,1,-1 do
    local input, output = getPredAndGTTables(DA[t], t)
    local dl = clonesD.criterion[t]:backward(input,output)
    local nGrad = #dl

    for dd=1,nGrad do
      table.insert(drnn_state[t], dl[dd]) -- gradient of loss at time t
    end

    local rnninp = getRNNDInput(t, rnn_state, predictions, rnn_stateE)    -- get combined RNN input table


    local dlst = clonesD.rnn[t]:backward(rnninp, drnn_state[t])

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
  end
  
  -- BACKWARDS ENCODER
  local drnn_stateE = {[TE] = drnn_state[0], true} -- true = zero the clones
  for t=TE,1,-1 do
    local rnninp = getRNNEInput(t, rnn_stateE)    -- get combined RNN input table
   
    local dlst = clonesE.rnn[t]:backward(rnninp, drnn_stateE[t])

    drnn_stateE[t-1] = {}
    local maxk = opt.num_layers+1
    if opt.model == 'lstm' then maxk = 2*opt.num_layers+1 end
    for k,v in pairs(dlst) do
      if k > 1 and k <= maxk then -- k == 1 is gradient on x, which we dont need
        -- note we do k-1 because first item is dembeddings, and then follow the
        -- derivatives of the state, starting at index 2. I know...
        drnn_stateE[t-1][k-1] = v
      end
    end
        
  end
  

  --   feval_mm = 0
  feval_multass = 0
--  print(grad_paramsE[1])
--  print(grad_paramsD[1])
--  grad_params = grad_paramsE:cat(grad_paramsD)
--  grad_params = grad_paramsD:clone()
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  return loss, grad_params

end




train_losses = {}
val_losses = {}
real_losses = {}
train_mm, val_mm, real_mm = {}, {}, {}
train_ma, val_ma, real_ma = {}, {}, {}
mot15_mota = {}
mot15_mota_nopos = {}
local optim_state = {learningRate = opt.lrng_rate, alpha = opt.decay_rate}
local glTimer = torch.Timer()
for i = 1, opt.max_epochs do
  local epoch = i
  globiter = i

  local timer = torch.Timer()

  local _, loss = optim.rmsprop(feval, params, optim_state)
  local time = timer:time().real

  local train_loss = loss[1] -- the loss is inside a list, pop it
  train_losses[i] = train_loss

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
      time, optim_state.learningRate, glTimer:time().real)
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
    val_loss = eval_val()
    val_losses[i] = math.max(val_loss, 1e-5)

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


    --       print(train_losses)
    --       print(plot_loss)
    local minTrainLoss, minTrainLossIt = torch.min(plot_loss,1)
    local minValidLoss, minValidLossIt = torch.min(plot_val_loss,1)
    local minRealLoss, minRealLossIt = torch.min(plot_real_loss,1)
    minTrainLoss=minTrainLoss:squeeze()
    minTrainLossIt=minTrainLossIt:squeeze()*opt.eval_val_every
    minValidLoss=minValidLoss:squeeze()
    minValidLossIt=minValidLossIt:squeeze()*opt.eval_val_every
    minRealLoss=minRealLoss:squeeze()
    minRealLossIt=minRealLossIt:squeeze()*opt.eval_val_every

    -- TODO there is a bug in best training loss and best training DA
    pm('--------------------------------------------------------')
    pm(string.format('%10s%10s%10s%10s','Losses','Training','Valid','Real'))
    pm(string.format('%10s%10.5f%10.5f%10.5f','Current',plot_loss[-1],val_loss,real_loss))
    pm(string.format('%10s%10.5f%10.5f%10.5f','Best',minTrainLoss,  minValidLoss, minRealLoss))
    pm(string.format('%10s%10d%10d%10d','Iter',minTrainLossIt,  minValidLossIt, minRealLossIt))


    local minTrainLoss, minTrainLossIt = torch.min(plot_train_mm,1)
    local minValidLoss, minValidLossIt = torch.min(plot_val_mm,1)
    local minRealLoss, minRealLossIt = torch.min(plot_real_mm,1)
    minTrainLoss=minTrainLoss:squeeze()      minTrainLossIt=minTrainLossIt:squeeze()*opt.eval_val_every
    minValidLoss=minValidLoss:squeeze()      minValidLossIt=minValidLossIt:squeeze()*opt.eval_val_every
    minRealLoss=minRealLoss:squeeze()        minRealLossIt=minRealLossIt:squeeze()*opt.eval_val_every

    pm('--------------------------------------------------------')
    pm(string.format('%10s%10s%10s%10s','MissDA','Training','Valid','Real'))
    pm(string.format('%10s%10.2f%10.2f%10.2f','Current',feval_mm+1,eval_val_mm+1,eval_benchmark_mm+1))
    pm(string.format('%10s%10.2f%10.2f%10.2f','Best',minTrainLoss,  minValidLoss, minRealLoss))
    pm(string.format('%10s%10d%10d%10d','Iter',minTrainLossIt,  minValidLossIt, minRealLossIt))
    pm('--------------------------------------------------------')

    -- save checkpt
    local savefile = getCheckptFilename(modelName, opt, modelParams)
    saveCheckpoint(savefile, tracks, detections, protosD, opt, train_losses, glTimer:time().real, i, protosE)


    --       os.execute("th rnnTrackerY.lua -model_name testingY -model_sign r50_l2_n3_m3_d2_b1_v0_li1 -suppress_x 0 -length 5 -seq_name TUD-Crossing")
    -- save lowest val if necessary
    if val_loss <= torch.min(plot_val_loss) then
      pm('*** New min. validation loss found! ***',2)
      local fn, dir, base, signature, ext = getCheckptFilename(modelName, opt, modelParams)
      local savefile  = dir .. base .. '_' .. signature .. '_val' .. ext
      saveCheckpoint(savefile, tracks, detections, protosD, opt, train_losses, glTimer:time().real, i, protosE)
    end

    -- plot
    local lossPlotTab = {}
    table.insert(lossPlotTab, {"Trng loss",plot_loss_x,plot_loss, 'with linespoints lt 1'})
    table.insert(lossPlotTab, {"Vald loss",plot_val_loss_x, plot_val_loss, 'with linespoints lt 3'})
    --       table.insert(lossPlotTab, {"Real loss",plot_real_loss_x, plot_real_loss, 'linespoints lt 5'})
    table.insert(lossPlotTab, {"Trng MM",plot_train_mm_x, plot_train_mm, 'with points lt 1'})
    table.insert(lossPlotTab, {"Vald MM",plot_val_mm_x, plot_val_mm, 'with points lt 3'})
    --       table.insert(lossPlotTab, {"Real MM",plot_real_mm_x, plot_real_mm, 'points lt 5'})

    --       table.insert(lossPlotTab, {"Trng MA",plot_train_ma_x, plot_train_ma, 'points pt 1'})
    --       table.insert(lossPlotTab, {"Vald MA",plot_val_ma_x, plot_val_ma, 'points pt 3'})


    --  local minInd = math.min(1,plot_loss:nElement())
    local maxY = math.max(torch.max(plot_loss), torch.max(plot_val_loss), torch.max(plot_real_loss),
      torch.max(plot_train_mm), torch.max(plot_val_mm), torch.max(plot_real_mm))*2
    local minY = math.min(torch.min(plot_loss), torch.min(plot_val_loss), torch.min(plot_real_loss),
      torch.min(plot_train_mm), torch.min(plot_val_mm), torch.min(plot_real_mm))/2
    rangeStr = string.format("set xrange [%d:%d];set yrange [%f:%f]",
      opt.eval_val_every-1, i+1, minY, maxY)
    --       rangeStr = string.format("set yrange [%f:%f]", minY, maxY)
    local rawStr = {}
    table.insert(rawStr, rangeStr)
    table.insert(rawStr, 'set logscale y')

    local winTitle = string.format('Loss-%06d-%06d',itOffset+1,opt.max_epochs+itOffset)
    plot(lossPlotTab, 2, winTitle, rawStr, 1) -- plot and save (true)
    gnuplot.raw('unset logscale') -- for other plots


    printModelOptions(opt, modelParams) -- print parameters
  end

  if (i == 1 or i % (opt.print_every*10) == 0) and i<opt.max_epochs then
    printModelOptions(opt, modelParams)
    printTrainingHeadline()
  end -- headline

end

print('-------------   PROFILING   INFO   ----------------')
print(string.format('%20s%10.2f%7s','total time',ggtime:time().real,''))