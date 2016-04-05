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

  --  oneCost[{{},{opt.max_m+1,opt.nClasses}}] = getFillMatrix(true, opt.miss_thr, opt.dummy_noise, 1)
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

--------------------------------------------------------------------------
--- Generate Data for Hungarian training
function genHunData(nSamples)
  --   if opt.permute ~= 0 then error('permutation not implemented') end
  local CostTab, HunTab = {},{}
  local missThr = opt.miss_thr
  for n=1,nSamples do

    if n%math.floor(nSamples/5)==0 then
      print((n)*(100/(nSamples))..' %...')
    end

    local oneCost = getOneCost()
    --    print(oneProb)
    --    local oneProb = makeProb(oneCost:reshape(opt.mini_batch_size*opt.max_n, opt.nClasses)):reshape(opt.mini_batch_size, opt.max_n * opt.nClasses)


    --     oneCost = oneCost:cat(torch.ones(miniBatchSize, maxTargets)*missThr)
    --     print(oneCost)
    --     abort()
    oneCost = dataToGPU(oneCost)
    table.insert(CostTab, oneCost)

    local hunSols = torch.ones(1,opt.max_n):int()
    for m=1,opt.mini_batch_size do
      local costMat = oneCost[m]:reshape(opt.max_n, opt.nClasses)
      local ass = hungarianL(costMat)
      hunSols = hunSols:cat(ass[{{},{2}}]:reshape(1,opt.max_n):int(), 1)
    end
    hunSols=hunSols:sub(2,-1)
    table.insert(HunTab, hunSols)
  end
  return CostTab, HunTab
end

--------------------------------------------------------------------------
--- Generate Data for marginals training
function genMarginalsData(nSamples)
  --   if opt.permute ~= 0 then error('permutation not implemented') end
  local ProbTab, HunTab = {},{}
  local missThr = opt.miss_thr


  --  print(solTable)
  --  abort()

  for n=1,nSamples do

    if n%math.floor(nSamples/5)==0 then
      print((n)*(100/(nSamples))..' %...')
    end

    --    if n==1 or opt.permute<=1 or (opt.permute>1 and n%opt.permute == 0) then
    --      oneCost = getOneCost()
    --    else
    ----       print('permuting?')
    --      oneCost = permuteCost(oneCost)
    --    end
    local randN = math.random(5)
    local randProb = torch.rand(opt.mini_batch_size, opt.max_n * opt.nClasses)
    --    print(randProb)
    --    abort()
    local oneProb = torch.pow(torch.abs(randProb ),randN)
    --    print(oneProb)
    oneProb = makeProb(oneProb:reshape(opt.mini_batch_size*opt.max_n, opt.nClasses)):reshape(opt.mini_batch_size, opt.max_n * opt.nClasses)
    --    print(oneProb)
    --    abort()

    --    print(oneCost)
    --    abort()

    oneProb = dataToGPU(oneProb)
    table.insert(ProbTab, oneProb)

    --    local hunSols = torch.ones(1,opt.max_n):int()
    --    for m=1,opt.mini_batch_size do
    --      local costmat = oneCost[m]:reshape(opt.max_n, opt.nClasses)
    --      local ass = hungarianL(costmat)
    --      hunSols = hunSols:cat(ass[{{},{2}}]:reshape(1,opt.max_n):int(), 1)
    --    end
    --    hunSols=hunSols:sub(2,-1)

    local hunSols = torch.ones(1,opt.max_n*opt.max_m):float()
    for m=1,opt.mini_batch_size do
      local probMat = oneProb[m]:reshape(opt.max_n, opt.nClasses)

      local mar = getMarginals(probMat:float(),solTable):reshape(1,opt.max_n*opt.max_m):float()
      --      print(hunSols)
      hunSols = hunSols:cat(mar  ,1)
    end
    hunSols=hunSols:sub(2,-1)

    hunSols = dataToGPU(hunSols)
    table.insert(HunTab, hunSols)
  end
  return ProbTab, HunTab
end


--------------------------------------------------------------------------
--- read QBP data and solutions
function readQBPData(ttmode)

  if ttmode ==nil or (ttmode~='train' and ttmode~='test') then ttmode = 'train' end

  local ProbTab, SolTab = {},{}
  local ValProbTab, ValSolTab = {},{}

  local Qfile = string.format('%sdata/%s/Q_N%d_M%d',getRootDir(), ttmode, opt.max_n, opt.max_m);
  checkFileExist(Qfile..'.mat','Q cost file')  
  local loaded = mattorch.load(Qfile..'.mat')
  local allQ = loaded.allQ:t() -- TODO why is it transposed?


  local Solfile = string.format('%sdata/%s/Sol_N%d_M%d',getRootDir(), ttmode, opt.max_n, opt.max_m);
  local allSol = {}
  if opt.solution == 'integer' then
    Solfile = string.format('%sdata/%s/SolInt_N%d_M%d',getRootDir(), ttmode, opt.max_n, opt.max_m);
    checkFileExist(Solfile..'.mat','solution file')
    local loaded = mattorch.load(Solfile..'.mat')
    allSol = loaded.allSolInt:t() -- TODO why is it transposed?
  elseif opt.solution == 'distribution' then
    Solfile = string.format('%sdata/%s/Sol_N%d_M%d',getRootDir(), ttmode, opt.max_n, opt.max_m);
    checkFileExist(Solfile..'.mat','solution file')
    local loaded = mattorch.load(Solfile..'.mat')
    allSol = loaded.allSol:t() -- TODO why is it transposed?
  end

  allQ=allQ:float()
  allSol=allSol:float()


  pm('Loaded data matrix of size '..allQ:size(1) .. ' x '..allQ:size(2))
  pm('Loaded soln matrix of size '..allSol:size(1) .. ' x '..allSol:size(2))

  local totalDataSamples = allQ:size(1)
  local trainSamplesNeeded = opt.synth_training*opt.mini_batch_size
  local validSamplesNeeded = opt.synth_valid*opt.mini_batch_size
  local totalSamplesNeeded = trainSamplesNeeded + validSamplesNeeded
  local maxTrainingSample = math.min(trainSamplesNeeded,totalDataSamples-validSamplesNeeded)
  pm('We will need ' .. trainSamplesNeeded .. ' training samples')
  pm('We will need ' .. validSamplesNeeded .. ' validation samples')
  pm(string.format('We will use %.1f %% of data for training', trainSamplesNeeded/totalDataSamples*100))

  if totalDataSamples < totalSamplesNeeded then
    if totalDataSamples < validSamplesNeeded then
      error('not enough data...')
    end
  end

--  allQ = dataToGPU(allQ)
--  allSol = dataToGPU(allSol)

  local nth = 0 -- counter for reading lines
  --  local solSize = opt.max_n*opt.max_m -- one hot
  --  local opt.solSize = opt.max_n -- integer

  -- training data
  pm('training data...')
  local nSamples = opt.synth_training
  for n=1,nSamples do
    if n%math.floor(nSamples/2)==0 then print((n)*(100/(nSamples))..' %...') end

    local oneBatch = torch.zeros(1,opt.inSize)
    local oneBatchSol = torch.zeros(1,opt.solSize)

    for mb=1,opt.mini_batch_size do
      nth=nth+1
      if nth>maxTrainingSample then nth=1 end
      oneBatch = oneBatch:cat(allQ[nth]:reshape(1,opt.inSize),1)
      oneBatchSol = oneBatchSol:cat(allSol[nth]:reshape(1,opt.solSize),1)
      --      oneBatchSol = oneBatchSol:cat(getMarginals(allQ[nth]:reshape(opt.max_n,opt.max_m):float(),solTable):reshape(1,opt.max_n*opt.max_m):float(),1)
    end
    oneBatch=oneBatch:sub(2,-1)
    oneBatchSol=oneBatchSol:sub(2,-1)
    oneBatch = dataToGPU(oneBatch)
    oneBatchSol = dataToGPU(oneBatchSol)
    table.insert(ProbTab, oneBatch)
    table.insert(SolTab, oneBatchSol)
  end

  pm('validation data...')
  -- validation data
  nth=maxTrainingSample
  local nSamples = opt.synth_valid
  for n=1,nSamples do
    if n%math.floor(nSamples/2)==0 then print((n)*(100/(nSamples))..' %...') end

    local oneBatch = torch.zeros(1,opt.inSize)
    local oneBatchSol = torch.zeros(1,opt.solSize)

    for mb=1,opt.mini_batch_size do
      nth=nth+1
      oneBatch = oneBatch:cat(allQ[nth]:reshape(1,opt.inSize),1)
      oneBatchSol = oneBatchSol:cat(allSol[nth]:reshape(1,opt.solSize),1)
      --      oneBatchSol = oneBatchSol:cat(getMarginals(allQ[nth]:reshape(opt.max_n,opt.max_m):float(),solTable):reshape(1,opt.max_n*opt.max_m):float(),1)
    end
    oneBatch=oneBatch:sub(2,-1)
    oneBatchSol=oneBatchSol:sub(2,-1)
    oneBatch = dataToGPU(oneBatch)
    oneBatchSol = dataToGPU(oneBatchSol)    
    table.insert(ValProbTab, oneBatch)
    table.insert(ValSolTab, oneBatchSol)
  end

  return ProbTab, SolTab, ValProbTab, ValSolTab
end

function computeMarginals(CostTab)
  local SolTab = {}
  
  local N,M = opt.max_n, opt.max_m
  for k,v in pairs(CostTab) do
  
    if k%math.floor(#CostTab/10)==0 then print((k)*(100/(#CostTab))..' %...') end
    
    local batchMarginals = torch.zeros(opt.mini_batch_size,N*M):float()
    for mb=1,opt.mini_batch_size do
      local C = dataToCPU(v:sub(mb,mb))
--      print(k,mb)
     
          
      local marginals = torch.zeros(N,M):double() -- double precision necessary here
      for key, var in pairs(solTable.feasSol) do
        local idx = var:eq(1)              -- find assignments
        local hypCost = 0
        if opt.problem=='linear' then 
          hypCost = evalSol(var, C)   -- get solution for one assignment hypothesis
        elseif opt.problem=='quadratic' then
          hypCost = evalSol(var, nil, C)   -- get solution for one assignment hypothesis
        end
--        print(hypCost)
--        print(torch.exp(-hypCost))
--        marginals[idx] = marginals[idx] + torch.exp(-hypCost)   -- add to joint matrix
          marginals[idx] = marginals[idx] + torch.exp(hypCost)   -- add to joint matrix
      end
      
--      print(marginals)
--      print('---')
--      print(C[1][1])
--      print(marginals[1][1])
--      local beforeProb = marginals:clone()
      marginals = makeProb(marginals)
--      print(marginals[1][1])
--      if marginals[1][1]~=marginals[1][1] then
--        print(k)
--        print(beforeProb)
--        print(torch.sum(beforeProb,2))
--        for i=1,N do
--          print(i)
--          print(beforeProb[i]) 
--          print(torch.sum(beforeProb[i]))
--          print(beforeProb[i]/torch.sum(beforeProb[i])) 
--        end
--        print(marginals)
--        abort()
--      end
--      sleep(.1)
      batchMarginals[mb] = marginals:reshape(1,N*M)
    end
    table.insert(SolTab, dataToGPU(batchMarginals))
  end
  
  
  return SolTab
end

function normalizeCost(CostTab)
  for k,v in pairs(CostTab) do
--    print(CostTab[k])
    for mb=1,opt.mini_batch_size do
      local oneCost = v[mb]:clone()
      local minv = torch.min(oneCost)
      oneCost = oneCost - minv
      local maxv = torch.max(oneCost)
      CostTab[k][mb] = oneCost / maxv
    end
--    print(CostTab[k])
--    abort()
    
  end
  return CostTab
end


function findFeasibleSolutions(N,M)
  pm(string.format('Finding all feasible %d x %d assignments...',N,M))
  -- using Penlight permute
  local feasSol = {}

  local permute = require 'pl.permute';

  local lin = torch.linspace(1,N,N):totable()
  local allPermutations = permute.table(lin)
  for k,v in pairs(allPermutations) do
    local ass = torch.zeros(N,M)
    for i=1,N do
      ass[i][v[i]]=1
    end
    table.insert(feasSol, ass)
  end


  solTable = {}
  solTable.feasSol = feasSol
  solTable.infSol = infSol

  print('Feasible solutions: '..#feasSol)
  --  abort()

  return solTable

end

function findFeasibleSolutions2(N,M)
  --  local assFile = 'tmp/ass_n'..N..'_m'..M..'.t7'

  pm('Generating feasible solutions table...')
  local feasSol, infSol = {}, {}
  local possibleAssignments = math.pow(2,N*M)

  for s=1,possibleAssignments do

    local binCode = torch.Tensor(toBits(s, N*M))
    local ass = binCode:reshape(N,M)
    local feasible = true
    local sumAllEntries = torch.sum(binCode)
    if sumAllEntries ~= N then goto continue end

    local sumOverColumns = torch.sum(ass,2)
    if torch.sum(sumOverColumns:ne(1)) > 0 then goto continue end

    local sumOverRows = torch.sum(ass,1)
    if torch.sum(sumOverRows:ne(1)) > 0 then goto continue end


    table.insert(feasSol, ass)

    --    print(s)
    ::continue::
  end
  solTable = {}
  solTable.feasSol = feasSol
  solTable.infSol = infSol
  --      torch.save(assFile, solTable)

  pm('... done')
  return solTable

end

--------------------------------------------------------------------------
--- get all inputs for one time step
-- @param t   time step
-- @param rnn_state hidden state of RNN (use t-1)
-- @param predictions current predictions to use for feed back
function getRNNInput(t, rnn_state, predictions)
  local rnninp = {}

  -- Cost matrix
  local loccost = costs:clone():reshape(opt.mini_batch_size, opt.inSize)
  --  loccost = probToCost(loccost)
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

function getDebugTableTitle(str)
  local formatString = string.format('%%%ds',opt.max_m * 8+1)
  return string.format(formatString, '--- '..str..' --- ')
end


--------------------------------------------------------------------------
--- print all values for looking at them :)
-- @param C     The cost matrix
-- @param Pred  Predicted values from LSTM
function printDebugValues(C, Pred)

  --  local C = probToCost(C) -- negative log-probability (cost)
  --  local N,M = getDataSize(P)
  C=dataToCPU(C)
  Pred=dataToCPU(Pred)
  local N,M = opt.max_n, opt.max_m

  --  print('Cost matrix')
  if opt.problem == 'linear' then
    local minv, mini = torch.min(C,2)
    minv=minv:reshape(N) mini=mini:reshape(N)

    local HunAss = hungarianL(C)
    local marginals = getMarginals(C,solTable)

    marginals = dataToCPU(marginals)
    local mmaxv, mmaxi = torch.max(marginals,2)
    mmaxv=mmaxv:reshape(N) mmaxi=mmaxi:reshape(N)

    local pmmaxv, pmmaxi = torch.max(Pred,2)
    pmmaxv=pmmaxv:reshape(N) pmmaxi=pmmaxi:reshape(N)



    print(string.format('%5s%5s%5s%5s%5s%5s|%s|%s|%s','i','NN','HA','Mar','PMar','Err',
      getDebugTableTitle('Cost'),getDebugTableTitle('Marg'),getDebugTableTitle('Predicted Marg')))
    for i=1,N do
      local prLine = ''
      prLine = prLine .. string.format('%5d%5d%5d%5d%5d%5d|',i,mini[i],HunAss[i][2],mmaxi[i],pmmaxi[i],mmaxi[i]-pmmaxi[i])
      for j=1,M do
        --      prLine = prLine ..  string.format('%8.4f',C[i][j])
        prLine = prLine ..  string.format('%8.4f',C[i][j])
      end
      prLine = prLine .. ' |'
      for j=1,M do
        prLine = prLine ..  string.format('%8.4f',marginals[i][j])
      end
      prLine = prLine .. ' |'
      for j=1,M do
        prLine = prLine ..  string.format('%8.4f',Pred[i][j])
      end

      --    prLine=prLine..'\n'
      print(prLine)
    end
    --  print(probMat)
    --  print(mini:long():reshape(N,1))
    local NNcost = torch.sum(C:gather(2,mini:long():reshape(N,1)))
    local HAcost = torch.sum(C:gather(2,HunAss:narrow(2,2,1):long():reshape(N,1)))
    local Marcost = torch.sum(C:gather(2,mmaxi:long():reshape(N,1)))
    local PMarcost = torch.sum(C:gather(2,pmmaxi:long():reshape(N,1)))

    local MMsum = torch.sum((mmaxi-pmmaxi):ne(0)) -- sum of wrong predictions
    --  print(NNcost,HAcost,Marcost,PMarcost)
    print(string.format('%5s%5.2f%5.2f%5.2f%5.2f%5d','Sum',NNcost,HAcost,Marcost,PMarcost,MMsum))
  elseif opt.problem=='quadratic' then
    if N<4 and M<4 then
      print('QBP')


      for i=1,N*M do
        local prLine = ''
        for j=1,N*M do prLine = prLine .. string.format('%8.4f ',C[i][j]) end
        print(prLine)
      end
    end

    local sol = huns:sub(1,1)
    if opt.solution=='integer' then
      sol=getOneHotLab(sol, true)
    else
      sol=sol:reshape(N,M)
    end
    
    sol=dataToCPU(sol)
    
    local diff = sol-Pred

    -- true solution
    local smaxv, smaxi = torch.max(sol,2)
    smaxv=smaxv:reshape(N) smaxi=smaxi:reshape(N)

    -- predicted solution
    local pmaxv, pmaxi = torch.max(Pred, 2)
    pmaxv=pmaxv:reshape(N) pmaxi=pmaxi:reshape(N)

    if opt.inference == 'map' then
    print(string.format('%5s%5s%5s%5s%5s|%s|%s|%s','i','Sol','Mar','Pred','Err',
      getDebugTableTitle('Diff'),getDebugTableTitle('Optim'),getDebugTableTitle('Predict')))
    elseif opt.inference=='marginal' then
      print(string.format('%5s%5s%5s%5s%5s|%s|%s|%s','i','Sol','Diff','Pred','Err',
        getDebugTableTitle('Diff'),getDebugTableTitle('Marginal'),getDebugTableTitle('Pred Marg')))
    end

    for i=1,N do
      local prLine = ''
      prLine = prLine .. string.format('%5d%5d%5.1f%5d%5d|',i,smaxi[i],torch.sum(torch.abs(diff[i])),pmaxi[i],smaxi[i]-pmaxi[i])
      for j=1,M do prLine = prLine ..  string.format('%8.4f',diff[i][j]) end
      prLine = prLine .. ' |'
      for j=1,M do prLine = prLine ..  string.format('%8.4f',sol[i][j]) end
      prLine = prLine .. ' |'
      for j=1,M do prLine = prLine ..  string.format('%8.4f',Pred[i][j]) end
      print(prLine)
    end

    local s = dataToCPU(getOneHotLab(pmaxi,true))
    local maxMargins = dataToCPU(getOneHotLab(pmaxi,true))
    local maxSol = sol:clone()
    if opt.solution == 'distribution' then  maxSol=dataToCPU(getOneHotLab(smaxi, true)) end


    local solProb = evalSol(maxSol,nil,C)
    local predProb = evalSol(maxMargins, nil, C)
    local MMsum = torch.sum((smaxi-pmaxi):ne(0)) -- sum of wrong predictions

    print(string.format('%5s%5.1f%5.1f%5.1f%5d|','Sim',solProb,torch.sum(torch.abs(diff)),predProb,MMsum))

  end


end

function createAuxDirs()
  local rootDir = getRootDir()
  mkdirP(rootDir..'/bin')
  mkdirP(rootDir..'/tmp')
  mkdirP(rootDir..'/out')
  mkdirP(rootDir..'/config')
  mkdirP(rootDir..'/graph')
  mkdirP(rootDir..'/data')
  mkdirP(rootDir..'/data/train')
  mkdirP(rootDir..'/data/test')
end

function evalSol(sol, c, Q)
  -- create a zero unary vector if not given
  if c==0 or c==nil then c=torch.zeros(1,opt.max_n*opt.max_m) end
  if Q==0 or Q==nil then Q=torch.zeros(opt.max_n*opt.max_m, opt.max_n*opt.max_m) end
  assert(c:nElement()==opt.max_n*opt.max_m, 'c not valid')
  assert(Q:nElement()==opt.max_n*opt.max_m*opt.max_n*opt.max_m, 'Q not valid')

  sol = sol:reshape(opt.max_n*opt.max_m,1)
  c = c:reshape(1,opt.max_n*opt.max_m)
  Q = Q:reshape(opt.max_n*opt.max_m,opt.max_n*opt.max_m)

  local ret = c * sol + sol:t() * Q * sol
  return ret:squeeze()
end

function plotProgress(predictions,winID, winTitle)
  local mm=0 -- number of mismatches
  local logpredDA = decode(predictions):reshape(opt.mini_batch_size*opt.max_n,opt.nClasses):sub(1,opt.max_n)
  local predDA=costToProb(-logpredDA)
  local predAsTracks = predDA:reshape(opt.max_n, opt.max_m, 1)
  
  local hun = huns:sub(1,1)

  eval_val_mm = 0
  
  local pmmaxv, pmmaxi = torch.max(predDA,2)
  local cmtr = costs:sub(1,1)
  --    print(cmtr)
  if opt.problem == 'linear' then cmtr = cmtr:reshape(opt.max_n,opt.max_m)
  elseif opt.problem == 'quadratic' then cmtr = cmtr:reshape(opt.max_n*opt.max_m,opt.max_n*opt.max_m)
  end

  printDebugValues(cmtr, predDA:reshape(opt.max_n, opt.max_m))

  local sol = nil
  if opt.inference == 'map' then
    -- greedy
    sol = hun:reshape(opt.max_n,1)
    if opt.solution == 'distribution' then
      local sv, si = torch.max(hun:reshape(opt.max_n, opt.max_m),2)
      sol = si:clone()
    end
    mm = torch.sum(torch.abs(sol:float()-pmmaxi:float()):ne(0))
  elseif opt.inference == 'marginal' then
    -- marginals
    --        local marginals = getMarginals(cmtr,solTable)
    sol = hun:reshape(opt.max_n, opt.max_m)
    local mmaxv, mmaxi = torch.max(sol,2)

    mm = torch.sum(torch.abs(mmaxi-pmmaxi):ne(0))
  end
  sol=dataToCPU(sol)
  predDA=dataToCPU(predDA)
  if opt.solution == 'integer' then sol=getOneHotLab(sol, true) end
  -- plot prob distributions
  local plotTab = {}
  gnuplot.raw('unset ytics')
  plotTab = getPredPlotTab(plotTab, predDA, 1)
  plotTab = getPredPlotTab(plotTab, sol, 2)
  plot(plotTab, winID, winTitle)
  gnuplot.raw('set ytics')

  
  return mm
end
