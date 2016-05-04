require 'torch'
require 'nn'
require 'nngraph'

function fixOpt(opt)
  if opt.sparse~=0 then error('sparse data not implemented') end
  if opt.sparse~=0 and opt.mini_batch_size>1 then
    error('batch not supported for sparse data')
  end


  opt.max_m = opt.max_n
  opt.model = string.lower(opt.model) -- make model name lower case
  
  -- convert integer indexes to string descriptions 
  if opt.order == 1 then opt.problem = 'linear' 
  elseif opt.order == 2 then opt.problem ='quadratic'
  end

  if opt.inf_index == 1 then opt.inference = 'map'
  elseif opt.inf_index == 2 then opt.inference = 'marginal'
  end
    
  if opt.sol_index == 1 then opt.solution = 'integer'
  elseif opt.sol_index == 2 then opt.solution = 'distribution'
  end

  opt.inference = string.lower(opt.inference)
  if string.find(opt.inference,'marg')~=nil then opt.inference='marginal' end

  assert(opt.problem=='linear' or opt.problem=='quadratic', 'unknown problem option')
  assert(opt.solution=='integer' or opt.solution=='distribution', 'unknown solution option')
  assert(opt.inference=='map' or opt.inference=='marginal', 'unknown inference option')

  opt.model_index=1
  if opt.model=='gru' then opt.model_index=2; end
  if opt.model=='rnn' then opt.model_index=3; end
  opt.nClasses = opt.max_m

  --  opt.splitInput = 0
  opt.inSize = opt.max_n * opt.nClasses -- input feature vector size (Linear Assignment)
  if opt.problem=='quadratic' then
    opt.inSize = opt.max_n*opt.max_m * opt.max_n*opt.max_m -- QBP
  end
  --  opt.splitInput = 1
  opt.inSize2 = opt.inSize

  -- partial input (row-by-row...)
  if opt.full_input == 1 then
    opt.inSize2 = opt.nClasses
    if opt.problem == 'quadratic' then opt.inSize2 = opt.max_n*opt.max_n*opt.max_m end
  end

  opt.solSize = opt.max_n -- integer
  if opt.solution == 'distribution' then
    opt.solSize = opt.max_n*opt.max_m -- one hot (or full)
  end
  if opt.supervised == 0 then opt.solSize =opt.max_n*opt.max_m end

  opt.featmat_n, opt.featmat_m = opt.max_n, opt.max_m
  if opt.problem == 'quadratic' then
    opt.featmat_n, opt.featmat_m = opt.max_n * opt.max_m, opt.max_n * opt.max_m
  end

  if opt.double_input ~= 0 then
    opt.inSize = opt.inSize*2
    opt.inSize2 = opt.inSize2*2
    --    opt.featmat_n = opt.featmat_n*2
    --    opt.featmat_m = opt.featmat_m*2
  end

  -- setting as integers
--  if opt.problem == 'linear' then opt.order = 1
--  elseif opt.problem=='quadratic' then opt.order = 2
--  end
--
--  if opt.solution == 'integer' then opt.sol_index = 1
--  elseif opt.solution == 'distribution' then opt.sol_index = 2
--  end
--
--  if opt.inference == 'map' then opt.inf_index = 1
--  elseif opt.inference == 'marginal' then opt.inf_index = 2
--  end


  return opt
end

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

  local oneCost = torch.rand(opt.mini_batch_size*opt.max_n, opt.max_m)
  --   local fillMatrix = torch.ones(opt.mini_batch_size*opt.max_n,opt.max_m) * opt.miss_thr
  --   if opt.dummy_noise ~= 0 then
  --     fillMatrix = fillMatrix + torch.rand(opt.mini_batch_size*opt.max_n,opt.max_m) * opt.dummy_noise
  --   end

  --  oneCost[{{},{opt.max_m+1,opt.nClasses}}] = getFillMatrix(true, opt.miss_thr, opt.dummy_noise, 1)
  oneCost=oneCost:reshape(opt.mini_batch_size, opt.max_n*opt.max_m)

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
    --    print(oneCost)


    -- sparsify
    local doSparsify = true
    local doSparsify = false

    if doSparsify then
      local dimSizes = torch.Tensor({opt.max_n, opt.max_m})
      for m=1,opt.mini_batch_size do
        for dim=1,2 do
          local dSize=dimSizes[dim]
          local remNPts = math.random(dSize-1)
          --        local remPts = torch.randperm(dSize):sub(1,remNPts)
          --        for r=1,remNPts do
          --          local remPts = remPts[r]
          --          oneCost[m]:view(opt.max_n,opt.max_m):narrow(dim,remPts,1):fill(0)
          --        end
          local remNPts = math.random(dSize)-1 -- remove [0,N-1] points
          if remNPts>0 then
            oneCost[m]:view(opt.max_n,opt.max_m):narrow(dim,dSize-remNPts+1,remNPts):fill(0)
          end
        end
        --      local remNPts = math.random(dSize-1)
      end
      --    print(oneCost)
      --    abort()
    end
    table.insert(CostTab, oneCost)

    local hunSols = torch.ones(opt.mini_batch_size,opt.max_n):int()
    --    if opt.supervised == 1 then
    hunSols = torch.ones(1,opt.max_n):int()
    for m=1,opt.mini_batch_size do
      local costMat = oneCost[m]:reshape(opt.max_n, opt.max_m)
      local ass = hungarianL(-costMat) -- treat as similarity
      hunSols = hunSols:cat(ass[{{},{2}}]:reshape(1,opt.max_n):int(), 1)
    end
    hunSols=hunSols:sub(2,-1)
    --    end

    hunSols = dataToGPU(hunSols)
    table.insert(HunTab, hunSols)
  end



  if opt.solution == 'distribution' then
    -- convert to one hot
    HunTab = oneHotAll(HunTab)
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
function readQBPData(ttmode, testfile, readnth)
  readnth = readnth or 1
  
  local ProbTab, SolTab = {},{}
  local ValProbTab, ValSolTab = {},{}

  
  local inSize = opt.inSize
  if opt.double_input ~= 0 then opt.inSize =opt.inSize/2 end

  if ttmode ==nil or (ttmode~='train' and ttmode~='test') then ttmode = 'train' end
  local dataDir = string.format('%sdata/%s/',getRootDir(), ttmode)

  local Qfile = ''
  if testfile ~= nil then Qfile = testfile
  else
    -- find data file
    local allDataFiles = {}
    for file in lfs.dir(dataDir) do
      local ff = dataDir..file
      local srchstr=string.format('QBP[-]%s_m%d_N%d_M%d[-]0503',opt.inference,opt.mbst,opt.max_n, opt.max_m)      
      if ff:find(srchstr) then
        table.insert(allDataFiles, ff)
      end
    end
    if allDataFiles == {} then error('no data found') end
    readnth = math.min(readnth, #allDataFiles)
    Qfile = allDataFiles[readnth]
  end
  
--  local Qfile = string.format('%s/QBP-%s_m%d_N%d_M%d.mat', 
--        dataDir, opt.inference,opt.mbst,opt.max_n, opt.max_m);

--  local Qfile = string.format('%sdata/%s/QBP_N%d_M%d.mat',getRootDir(), ttmode, opt.max_n, opt.max_m);
  checkFileExist(Qfile,'Q cost file')
  local loaded = mattorch.load(Qfile)
  pm('Loaded data file '..Qfile)
  local allQ = loaded.allQ:t() -- transpose because Matlab is first-dim-major (https://groups.google.com/forum/#!topic/torch7/qDIoWnJzkcU)
  pm('Check first entry: '..allQ[1][1])
  local allSparseQ = nil
  if opt.sparse ~= 0 then allSparseQ = loaded.allSparseQ:t():float() end


  local Solfile = Qfile  
  local allSol = {}

  if opt.inference == 'map' then
    if opt.solution == 'integer' then
--      Solfile = string.format('%sdata/%s/QBP_N%d_M%d.mat',getRootDir(), ttmode, opt.max_n, opt.max_m);
      checkFileExist(Solfile,'solution file')
      local loaded = mattorch.load(Solfile)
      allSol = loaded.allSolInt:t() -- transpose because Matlab is first-dim-major
    elseif opt.solution == 'distribution' then
--      Solfile = string.format('%sdata/%s/QBP_N%d_M%d.mat',getRootDir(), ttmode, opt.max_n, opt.max_m);
      checkFileExist(Solfile,'solution file')
      local loaded = mattorch.load(Solfile)
      allSol = loaded.allSol:t() -- transpose because Matlab is first-dim-major
    end
  elseif opt.inference == 'marginal' then
--    Solfile = string.format('%sdata/%s/QBP_N%d_M%d.mat',getRootDir(), ttmode, opt.max_n, opt.max_m);
    checkFileExist(Solfile,'solution file')
    local loaded = mattorch.load(Solfile)
    allSol = loaded.allMarginals:t() -- transpose because Matlab is first-dim-major
    --    print(allSol[1]:sub(1,7))
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
--            print(allSol[nth])
--        print(allQ[nth])
--        print(opt.inSize)
--            abort()
      --      print(allSparseQ[nth])
      --      sleep(1)
      local oneQ = allQ[nth]:reshape(1,opt.inSize),1
      oneBatch = oneBatch:cat(allQ[nth]:reshape(1,opt.inSize),1)
      oneBatchSol = oneBatchSol:cat(allSol[nth]:reshape(1,opt.solSize),1)
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

  opt.inSize = inSize
  return ProbTab, SolTab, ValProbTab, ValSolTab
end

--------------------------------------------------------------------------
--- read QBP data and solutions
function readQBPallMProposals(ttmode, mBst, readnth)
  readnth = readnth or 1
  local SolTab = {},{}
  local ValSolTab = {},{}

  local inSize = opt.inSize
  if opt.double_input ~= 0 then opt.inSize =opt.inSize/2 end

  if ttmode ==nil or (ttmode~='train' and ttmode~='test') then ttmode = 'train' end
  local dataDir = string.format('%sdata/%s/',getRootDir(), ttmode)
  



  local Solfile = ''
  -- find data file
  local allDataFiles = {}
  for file in lfs.dir(dataDir) do
    local ff = dataDir..file
    local srchstr=string.format('QBP[-]%s_m%d_N%d_M%d[-]0503',opt.inference,opt.mbst,opt.max_n, opt.max_m)      
    if ff:find(srchstr) then
      table.insert(allDataFiles, ff)
    end
  end
  if allDataFiles == {} then error('no data found') end
  readnth = math.min(readnth, #allDataFiles)
  Solfile = allDataFiles[readnth]
--  local Solfile = string.format('%sdata/%s/QBP-%s_m%d_N%d_M%d.mat', 
--        getRootDir(), ttmode, opt.inference,opt.mbst,opt.max_n, opt.max_m);
  
        
          
  local allSol = {}

    checkFileExist(Solfile,'solution file')
    local loaded = mattorch.load(Solfile)
    pm('Loaded data file '..Solfile)
    --    allSol = loaded.allMarginals:t() -- transpose because Matlab is first-dim-major
    allSol = nil
    --    print(allSol[1]:sub(1,7))
    if mBst ~= nil then
--      local mBstField = string.format('all_%d_BestMarginals',mBst)
      local mBstField = nil
      if opt.inference == 'marginal' then
        mBstField = string.format('all_%d_BestMarginals',mBst)
      elseif opt.inference == 'map' then
        if opt.solution == 'distribution' then
          mBstField = string.format('all_%d_Proposals',mBst)
        elseif opt.solution == 'integer' then
          mBstField = string.format('all_%d_ProposalsInt',mBst)
        end
      end
      pm('loading var '..mBstField)
      allSol = loaded[mBstField]:t() -- transpose because Matlab is first-dim-major
      --      print(mBst)
      --      print(allSol[1])
      --      abort()
      --      print(allSol[1]:sub(1,7))
      --      abort()
      -- transpose because Matlab is first-dim-major
    else error('ha?')
    end
--  end

  local totalDataSamples = allSol:size(1)
  local trainSamplesNeeded = opt.synth_training*opt.mini_batch_size
  local validSamplesNeeded = opt.synth_valid*opt.mini_batch_size
  local totalSamplesNeeded = trainSamplesNeeded + validSamplesNeeded
  local maxTrainingSample = math.min(trainSamplesNeeded,totalDataSamples-validSamplesNeeded)

  allSol=allSol:float()

--  pm('Loaded soln matrix of size '..allSol:size(1) .. ' x '..allSol:size(2))

  --  allQ = dataToGPU(allQ)
  --  allSol = dataToGPU(allSol)

  local nth = 0 -- counter for reading lines
  --  local solSize = opt.max_n*opt.max_m -- one hot
  --  local opt.solSize = opt.max_n -- integer

  -- training data
  --  pm('training data...')
  local nSamples = opt.synth_training
  for n=1,nSamples do
    --    if n%math.floor(nSamples/2)==0 then print((n)*(100/(nSamples))..' %...') end

    local oneBatchSol = torch.zeros(1,opt.solSize)

    for mb=1,opt.mini_batch_size do
      nth=nth+1
      if nth>maxTrainingSample then nth=1 end
      oneBatchSol = oneBatchSol:cat(allSol[nth]:reshape(1,opt.solSize),1)
    end
    oneBatchSol=oneBatchSol:sub(2,-1)
    oneBatchSol = dataToGPU(oneBatchSol)
    table.insert(SolTab, oneBatchSol)
  end

  --  pm('validation data...')
  -- validation data
  nth=maxTrainingSample
  local nSamples = opt.synth_valid
  for n=1,nSamples do
    --    if n%math.floor(nSamples/2)==0 then print((n)*(100/(nSamples))..' %...') end

    local oneBatchSol = torch.zeros(1,opt.solSize)

    for mb=1,opt.mini_batch_size do
      nth=nth+1
      oneBatchSol = oneBatchSol:cat(allSol[nth]:reshape(1,opt.solSize),1)
    end
    oneBatchSol=oneBatchSol:sub(2,-1)
    oneBatchSol = dataToGPU(oneBatchSol)
    table.insert(ValSolTab, oneBatchSol)
  end

  opt.inSize = inSize
  --  print(SolTab[1])
  --  abort()
  return SolTab, ValSolTab
end


function computeMarginals(CostTab)
  local SolTab = {}
  local N,M = opt.max_n, opt.max_m

  -- try to load first
  local shaKey = getSha(CostTab)
  local marFile = getRootDir()..'/data/marginals/mar_N'..opt.max_n..'_M'..opt.max_m..'_'..shaKey..'t7'
  if exist(marFile) then
    pm('Load precomputed marginals.')
    local loaded = torch.load(marFile)
    return loaded
  end


  for k,v in pairs(CostTab) do

    if k%math.floor(#CostTab/10)==0 then print((k)*(100/(#CostTab))..' %...') end

    local batchMarginals = torch.zeros(opt.mini_batch_size,N*M):float()
    for mb=1,opt.mini_batch_size do
      local C = dataToCPU(v:sub(mb,mb))
      --      print(k,mb)


      local marginals = torch.zeros(N,M):double() -- double precision necessary here
      local permCnt=0
      for key, var in pairs(solTable.feasSol) do
        permCnt=permCnt+1
        local idx = var:eq(1)              -- find assignments
        local hypCost = 0
        if opt.problem=='linear' then
          hypCost = evalSol(var, C)   -- get solution for one assignment hypothesis
        elseif opt.problem=='quadratic' then
          hypCost = evalSol(var, nil, C)   -- get solution for one assignment hypothesis
        end
        if opt.exp_cost ~= 0 then hypCost = torch.exp(hypCost) end
        --        print(hypCost)
        --        print(torch.exp(-hypCost))
        --        marginals[idx] = marginals[idx] + torch.exp(-hypCost)   -- add to joint matrix
        --          marginals[idx] = marginals[idx] + torch.exp(hypCost)   -- add to joint matrix
        marginals[idx] = marginals[idx] + hypCost   -- add to joint matrix
        --          print(torch.sum(marginals))
        --        print(permCnt)
        --        print(marginals)
        --        sleep(1)
        --        if permCnt>=4 then break end

      end
      --      abort()

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

  if not onNetwork() and TRAINING then pm('Save marginals to disk') torch.save(marFile, SolTab) end

  return SolTab
end

function getData(opt, getTraining, getValidation, readnth)
  readnth = readnth or 1
  local _
  pm('getting training/validation data...')
  if opt.problem == 'linear' then
    ----- gen data for Hungarian
    if getTraining then TrCostTab,TrSolTab = genHunData(opt.synth_training) end
    if getValidation then ValCostTab,ValSolTab = genHunData(opt.synth_valid) end
  elseif opt.problem == 'quadratic'  then
    if getTraining and getValidation then
      TrCostTab,TrSolTab,ValCostTab,ValSolTab = readQBPData('train', nil, readnth)
      -- mbest marginals
      if opt.mbst > 0 then
        --      if opt.solution == 'marginal' then
        TrSolTab_m_Prop, ValSolTab_m_Prop = {}, {}
        for m = 1,opt.mbst do
          TrSolTab_m_Prop[m], ValSolTab_m_Prop[m] = {}, {}
          TrSolTab_m_Prop[m], ValSolTab_m_Prop[m] =  readQBPallMProposals('train',m, readnth)
        end
      end
    elseif getTraining then
      TrCostTab,TrSolTab = readQBPData('train', nil, readnth)
      if opt.mbst > 0 then
        --      if opt.solution == 'marginal' then
        TrSolTab_m_Prop = {}
        for m = 1,opt.mbst do
          TrSolTab_m_Prop[m] = {}
          TrSolTab_m_Prop[m] =  readQBPallMProposals('train',m, readnth)
        end
      end      
    end
  end

  if opt.inference == 'marginal' and opt.problem=='linear' then
    pm('Computing marginals...')
    if getTraining then TrSolTab = computeMarginals(TrCostTab) end
    if getValidation then ValSolTab = computeMarginals(ValCostTab) end
  end

  if getTraining then TrCostTab = prepData(TrCostTab) end
  if getValidation then ValCostTab = prepData(ValCostTab) end

  --  print(TrSolTab[1])
  if opt.project_valid ~= 0 then
    TrSolTab = projectValid(TrSolTab)
    ValSolTab = projectValid(ValSolTab)
  end
  --  print(TrSolTab[1])
  --  abort()

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

-- insert tokens into costs
function insertTokens(tab)
  local extraColumn = torch.ones(opt.featmat_n,1) * -1
  local TrCostTabT = {}
  for k,v in pairs(tab) do
    local newv = torch.zeros(opt.mini_batch_size, opt.inSize + opt.featmat_n)
    for mb=1,opt.mini_batch_size do
      local c = dataToCPU(v[mb])
      c = c:reshape(opt.featmat_n, opt.featmat_m)
      c = c:cat(extraColumn, 2):reshape(1,opt.inSize + opt.featmat_n)
      newv[mb] = c
    end
    TrCostTabT[k] = dataToGPU(newv)
  end
  return TrCostTabT
end


function projectValid(tab)
  for k,v in pairs(tab) do
    local projSol = v:clone()
    for m=1,opt.mini_batch_size do
      local ass = hungarianL(-v[m]:reshape(opt.max_n, opt.max_m)):narrow(2,2,1):t() -- negative for maximising similarity
      projSol[m] = getOneHotLab(ass,true,opt.max_n)
    end
    tab[k] = projSol:clone()
  end
  return tab
end


function prepData(tab)
  pm('normalizing...')
  tab = normalizeCost(tab)



  --  if opt.double_input ~= 0 then
  --    for k,v in pairs(tab) do tab[k] = v:cat(v,2) end
  --  end

  if opt.invert_input ~= 0 then
    local invInd = torch.linspace(tab[1]:size(2),1,tab[1]:size(2)):long()
    for k,v in pairs(tab) do tab[k] = v:index(2,invInd) end
  end

  return tab
end

function oneHotAll(tab)
  local newTab = {}
  local featLength = tab[1]:size(2) *tab[1]:size(2)
  --  print(tab[1])
  --  print(featLength)
  for k,v in pairs(tab) do
    local oneHots = torch.zeros(opt.mini_batch_size, featLength)
    for m=1,opt.mini_batch_size do
      local oneHot = dataToCPU(getOneHotLab(dataToCPU(v[m]), true, opt.max_m))
      --      print(oneHot)
      --      abort()
      --      print(TrSolTab[k][m])
      --      print(oneHot:reshape(1,opt.featmat_n*opt.featmat_m))
      oneHots[m] = oneHot
    end
    newTab[k] = dataToGPU(oneHots)
  end

  return newTab
end



function findFeasibleSolutions(N,M)
  if N>9 or M>9 then
    pm('Problem too large for exhaustive search')
    return {}
  end
  pm(string.format('Finding all feasible %d x %d assignments...',N,M))
  local feasSolFile = string.format('%stmp/feasSol_n%d_m%d.t7',getRootDir(), N, M)
  if exist(feasSolFile) then
    pm('Load precomputed solution permutations.')
    local loaded = torch.load(feasSolFile)
    return loaded
  end

  local feasSol = {}

  -- using Penlight permute
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
  --  solTable.infSol = infSol

  print('Feasible solutions: '..#feasSol)
  if not onNetwork() then pm('Save permutations to disk') torch.save(feasSolFile, solTable) end
  --  abort()

  return solTable

end

--function findFeasibleSolutions2(N,M)
--  --  local assFile = 'tmp/ass_n'..N..'_m'..M..'.t7'
--
--  pm('Generating feasible solutions table...')
--  local feasSol, infSol = {}, {}
--  local possibleAssignments = math.pow(2,N*M)
--
--  for s=1,possibleAssignments do
--
--    local binCode = torch.Tensor(toBits(s, N*M))
--    local ass = binCode:reshape(N,M)
--    local feasible = true
--    local sumAllEntries = torch.sum(binCode)
--    if sumAllEntries ~= N then goto continue end
--
--    local sumOverColumns = torch.sum(ass,2)
--    if torch.sum(sumOverColumns:ne(1)) > 0 then goto continue end
--
--    local sumOverRows = torch.sum(ass,1)
--    if torch.sum(sumOverRows:ne(1)) > 0 then goto continue end
--
--
--    table.insert(feasSol, ass)
--
--    --    print(s)
--    ::continue::
--  end
--  solTable = {}
--  solTable.feasSol = feasSol
--  solTable.infSol = infSol
--  --      torch.save(assFile, solTable)
--
--  pm('... done')
--  return solTable
--
--end

--------------------------------------------------------------------------
--- get all inputs for one time step
-- @param t   time step
-- @param rnn_state hidden state of RNN (use t-1)
-- @param predictions current predictions to use for feed back
function getRNNInput(t, rnn_state, predictions)
  local rnninp = {}

  -- Cost matrix
  local loccost = nil

  if opt.double_input ~= 0 then
    loccost = costs:reshape(opt.mini_batch_size,opt.inSize/2,1):expand(opt.mini_batch_size,opt.inSize/2,2):transpose(2,3):reshape(opt.mini_batch_size,opt.inSize)
  else
    loccost = costs:clone():reshape(opt.mini_batch_size, opt.inSize)
  end

  --  print(opt.inSize, opt.inSize2)
  --  print(loccost)
  --  if opt.supervised == 0 then loccost = dataToGPU(torch.ones(loccost:size())) - loccost end
  if opt.inSize2~=opt.inSize then
    --    print(t)
    --    print(loccost)
    local from = (t-1)*opt.inSize2+1
    --    print(t,from,opt.inSize, opt.inSize2)
    --    print(loccost:narrow(2,from,opt.inSize2))

    loccost = loccost:narrow(2,from,opt.inSize2)

    --    abort()
  end
  --  print(costs:size())
  --  print(loccost:size())
  --  print(loccost)
  --  print(opt.inSize)
  --  print(loccost:size())
  --  print(loccost)
  --  abort()
  --  loccost = probToCost(loccost)
  table.insert(rnninp, loccost)

  for i = 1,#rnn_state[t-1] do table.insert(rnninp,rnn_state[t-1][i]) end

  return rnninp, rnn_state
end

--------------------------------------------------------------------------
--- get one input at a time
-- @param t   time step
-- @param rnn_state hidden state of RNN (use t-1)
function getRNNEInput(t, rnn_state)
  local rnninp = {}

  -- Cost matrix one entry at a time
  local loccost = costsT:narrow(2,t,1)
  table.insert(rnninp, loccost)

  for i = 1,#rnn_state[t-1] do table.insert(rnninp,rnn_state[t-1][i]) end

  return rnninp, rnn_state
end

--------------------------------------------------------------------------
--- get decoder input
-- @param t   time step
-- @param rnn_state hidden state of RNN (use t-1)
-- @param predictions current predictions to use for feed back
function getRNNDInput(t, rnn_state, predictions, rnn_stateE)
  local rnninp = {}

  -- Cost matrix
  local loccost = dataToGPU(torch.zeros(opt.mini_batch_size, 1))
  --  loccost = probToCost(loccost)
  table.insert(rnninp, loccost)

  for i = 1,#rnn_state[t-1] do table.insert(rnninp,rnn_state[t-1][i]) end
  --  table.insert(rnninp, rnn_stateE)

  return rnninp, rnn_state
end

--------------------------------------------------------------------------
--- RNN decoder
-- @param predictions current predictions to use for feed back
-- @param t   time step (nil to predict for entire sequence)
function decode(predictions, tar)
  local DA, sol, c1, c2 = {}, {}, {}, {}
  local T = tabLen(predictions) -- how many targets
  local stepSolSize = opt.nClasses
  if opt.supervised == 0 then stepSolSize = 1 end
  if tar ~= nil then
    local lst = predictions[tar]
    --    print(lst)
    --    print(stepSolSize)
    --    print(opt.outSize)
    DA = lst[opt.daPredIndex]:reshape(opt.mini_batch_size, stepSolSize) -- opt.mini_batch_size*opt.max_n x opt.max_m
    if opt.supervised == 0 then
      sol = lst[opt.solPredIndex]:reshape(opt.mini_batch_size, opt.solSize)
      c1 = lst[opt.c1PredIndex]:reshape(opt.mini_batch_size, opt.c1Size)
      c2 = lst[opt.c2PredIndex]:reshape(opt.mini_batch_size, opt.c2Size)
    end
  else
    DA = zeroTensor3(opt.mini_batch_size,T,stepSolSize)
    if supervised == 0 then
      sol = zeroTensor3(opt.mini_batch_size,T,opt.solSize)
      c1 = zeroTensor3(opt.mini_batch_size,T,opt.c1Size)
      c2 = zeroTensor3(opt.mini_batch_size,T,opt.c2Size)
    end
    for tt=1,T do
      local lst = predictions[tt]
      DA[{{},{tt},{}}] = lst[opt.daPredIndex]:reshape(opt.mini_batch_size, 1, stepSolSize)
      if opt.supervised == 0 then
        sol[{{},{tt},{}}] = lst[opt.solPredIndex]:reshape(opt.mini_batch_size, 1, opt.solSize)
        c1[{{},{tt},{}}] = lst[opt.c1PredIndex]:reshape(opt.mini_batch_size, 1, opt.c1Size)
        c2[{{},{tt},{}}] = lst[opt.c2PredIndex]:reshape(opt.mini_batch_size, 1, opt.c2Size)
      end
    end
    --    print(DA)
    if opt.project_valid ~= 0 then
      local projSol = DA:clone()
      for m=1,opt.mini_batch_size do
        local ass = hungarianL(-DA[m]):narrow(2,2,1):t()
        projSol[m] = getOneHotLab(ass,true,opt.max_n)
      end
      DA = projSol:clone()
    end
    --    print(DA)
    --    abort()
  end
  return DA, sol, c1, c2
end

function getDebugTableTitle(str)
  if opt.max_m>10 then return str end
  local formatString = string.format('%%%ds',opt.max_m * 5+1)
  return string.format(formatString, '--- '..str..' --- ')
end


--------------------------------------------------------------------------
--- print all values for looking at them :)
-- @param C     The cost matrix
-- @param Pred  Predicted values from LSTM
function printDebugValues(C, Pred)
  local N,M = opt.max_n, opt.max_m
  local printMatrix = true
  if N>10 or M>10 then printMatrix=false end

  --  local C = probToCost(C) -- negative log-probability (cost)
  --  local N,M = getDataSize(P)

  C=dataToCPU(C)
  Pred=dataToCPU(Pred)



  -- sol is the solution matrix (binary or distribution)
  local sol = huns:sub(1,1)
  --  print(sol)
  if opt.solution=='integer' then sol=getOneHotLab(sol, true)
  else
    --    print(sol)
    sol=sol:reshape(N,M)
  end
  sol=dataToCPU(sol)

  -- diff matrix sol and prediction
  local diff = sol-Pred

  -- true assignment solution (max row-wise)
  local smaxv, smaxi = torch.max(sol,2)
  smaxv=smaxv:reshape(N) smaxi=smaxi:reshape(N)

  -- predicted assignment solution
  local pmaxv, pmaxi = torch.max(Pred, 2)
  pmaxv=pmaxv:reshape(N) pmaxi=pmaxi:reshape(N)

  if opt.problem == 'linear' then
    local minv, mini = torch.max(C,2)
    minv=minv:reshape(N) mini=mini:reshape(N)

    local HunAss = hungarianL(-C)  -- negative is for similarity

    --    local marginals = getMarginals(C,solTable)
    --
    --    marginals = dataToCPU(marginals)
    --    local smaxv, smaxi = torch.max(marginals,2)
    --    smaxv=smaxv:reshape(N) smaxi=smaxi:reshape(N)
    --
    --    local pmaxv, pmaxi = torch.max(Pred,2)
    --    pmaxv=pmaxv:reshape(N) pmaxi=pmaxi:reshape(N)



    print(string.format('%5s%5s%5s%5s%5s%5s|%s|%s|%s','i','NN','HA','GT','Pred','Err',
      getDebugTableTitle('Sim-ty'),getDebugTableTitle('GT'),getDebugTableTitle('Predicted')))
    for i=1,N do
      local prLine = ''
      prLine = prLine .. string.format('%5d%5d%5d%5d%5d%5d|',i,mini[i],HunAss[i][2],smaxi[i],pmaxi[i],smaxi[i]-pmaxi[i])
      if printMatrix then
        for j=1,M do
          --      prLine = prLine ..  string.format('%8.4f',C[i][j])
          prLine = prLine ..  string.format('%5.2f',C[i][j])
        end
        prLine = prLine .. ' |'
        for j=1,M do
          if opt.solution=='integer' then prLine = prLine ..  string.format('%5d',sol[i][j])
          else prLine = prLine ..  string.format('%5.2f',sol[i][j])
          end
        end
        prLine = prLine .. ' |'
        for j=1,M do
          prLine = prLine ..  string.format('%5.2f',Pred[i][j])
        end
      end
      --    prLine=prLine..'\n'
      print(prLine)
    end
    --  print(probMat)
    --  print(mini:long():reshape(N,1))
    local NNcost = torch.sum(C:gather(2,mini:long():reshape(N,1)))
    local HAcost = torch.sum(C:gather(2,HunAss:narrow(2,2,1):long():reshape(N,1)))
    local Marcost = torch.sum(C:gather(2,smaxi:long():reshape(N,1)))
    local PMarcost = torch.sum(C:gather(2,pmaxi:long():reshape(N,1)))

    local MMsum = torch.sum((smaxi-pmaxi):ne(0)) -- sum of wrong predictions
    --  print(NNcost,HAcost,Marcost,PMarcost)
    print(string.format('%5s%5.1f%5.1f%5.1f%5.1f%5d|','Sum',NNcost,HAcost,Marcost,PMarcost,MMsum))
  elseif opt.problem=='quadratic' then
    if N<2 and M<2 then
      print('QBP')


      for i=1,N*M do
        local prLine = ''
        for j=1,N*M do prLine = prLine .. string.format('%8.4f ',C[i][j]) end
        print(prLine)
      end
    end



    print(string.format('%5s%5s%5s%5s%5s|%s|%s|%s','i','Sol','Mar','Pred','Err',
      getDebugTableTitle('Diff'),getDebugTableTitle('GT'),getDebugTableTitle('Predict')))


    for i=1,N do
      local prLine = ''
      prLine = prLine .. string.format('%5d%5d%5.1f%5d%5d|',i,smaxi[i],torch.sum(torch.abs(diff[i])),pmaxi[i],smaxi[i]-pmaxi[i])
      for j=1,M do prLine = prLine ..  string.format('%5.2f',diff[i][j]) end
      prLine = prLine .. ' |'
      for j=1,M do
        if opt.solution=='integer' then prLine = prLine ..  string.format('%5d',sol[i][j])
        else prLine = prLine ..  string.format('%5.2f',sol[i][j])
        end
      end
      prLine = prLine .. ' |'
      for j=1,M do prLine = prLine ..  string.format('%5.2f',Pred[i][j]) end
      print(prLine)
    end

    local s = dataToCPU(getOneHotLab(pmaxi,true))
    local maxMargins = dataToCPU(getOneHotLab(pmaxi,true))
    local maxSol = sol:clone()
    if opt.solution == 'distribution' then  maxSol=dataToCPU(getOneHotLab(smaxi, true)) end


    local solProb = evalSol(maxSol,nil,C)
    local predProb = evalSol(maxMargins, nil, C)
    local MMsum = torch.sum((smaxi-pmaxi):ne(0)) -- sum of wrong predictions
    --    print(s)
    --    print(torch.sum(s,1))

    local asssum = torch.sum(s,1)
    local findMultiAss = asssum:gt(1)
    local sumMultiAss = asssum[findMultiAss] - 1
    local mmass, notass = torch.sum(sumMultiAss), torch.sum(asssum:eq(0))
    print(string.format('%5s%5.1f%5.1f%5.1f%5d| Multi-assign: %d, not assigned: %d','Sim',solProb,torch.sum(torch.abs(diff)),predProb,MMsum,mmass,notass))

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
  mkdirP(rootDir..'/logs')
  mkdirP(rootDir..'/data/train')
  mkdirP(rootDir..'/data/test')
  mkdirP(rootDir..'/data/marginals')
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

function evalBatchConstraints(sol)
  local c1 = torch.sum(sol,2) -- vertical sum
  local c1cost = torch.sum(c1:ne(1), 3)
  if c1cost:type()=='torch.ByteTensor' then c1cost = dataToGPU(c1cost) end
  c1cost = c1cost:reshape(opt.mini_batch_size, 1)

  -- this one is superfluous
  --  local c2 = torch.sum(sol,3) -- horizontal sum
  --  local c2cost = torch.sum(c2:ne(1) * opt.mu, 2)

  --  print(c1)
  --  print(c1cost)
  --  print(c2)
  --  print(c2cost)
  --  abort()
  return c1cost
end

function evalBatchEnergy(sol, Q)
  local obj = 1
  --  print(Q)
  --  print(sol)
  if opt.problem == 'linear' then
    obj = torch.bmm(Q,sol)
  elseif opt.problem == 'quadratic' then
    obj = torch.bmm(Q,sol)
    obj = torch.bmm(sol:transpose(2,3), obj)
  end
  return obj
end

function plotProgress(predictions,winID, winTitle, save)
  save = save or 0
  local mm=0 -- number of mismatches
  --  print(opt.daPredIndex)
  local predDA = nil
  if opt.supervised == 0 then
    predDA = predictions[1][opt.daPredIndex+1]:reshape(opt.mini_batch_size*opt.max_n,opt.nClasses):sub(1,opt.max_n)
  else
    predDA = decode(predictions):reshape(opt.mini_batch_size*opt.max_n,opt.nClasses):sub(1,opt.max_n)
    if opt.project_valid == 0 then predDA=costToProb(-predDA) end
  end
  local predAsTracks = predDA:reshape(opt.max_n, opt.max_m, 1)

  --  print(huns)
  local hun = huns:sub(1,1)

  eval_val_mm = 0

  local pmmaxv, pmmaxi = torch.max(predDA,2)
  local cmtr = costs:sub(1,1)
  if opt.double_input ~= 0 then cmtr=cmtr:sub(1,1,1,opt.inSize/2) end
  --      print(cmtr)
  --      abort()
  if opt.problem == 'linear' then cmtr = cmtr:reshape(opt.max_n,opt.max_m)
  elseif opt.problem == 'quadratic' then cmtr = cmtr:reshape(opt.max_n*opt.max_m,opt.max_n*opt.max_m)
  end

  printDebugValues(cmtr, predDA:reshape(opt.max_n, opt.max_m))

  local sol = nil
  if opt.inference == 'map' then
    -- greedy

    if opt.solution == 'integer' then
      sol = hun:reshape(opt.max_n,1)
    elseif opt.solution == 'distribution' then
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
  --  print(sol)
  --  if opt.solution == 'integer' then
  if opt.inference ~= 'marginal' then sol=getOneHotLab(sol, true) end
  --  end
  -- plot prob distributions
  if winTitle ~= '' then
    local plotTab = {}
    gnuplot.raw('unset ytics')
    plotTab = getPredPlotTab(plotTab, predDA, 1)
    plotTab = getPredPlotTab(plotTab, sol, 2)
    plot(plotTab, winID, winTitle, {}, save)
    gnuplot.raw('set ytics')
  end

  return mm
end

function plotProgressD(predictions,winID, winTitle)
  local mm=0 -- number of mismatches
  local decodePred = decode(predictions)
  local logpredDA = decodePred:sub(1,1):reshape(opt.max_n,opt.max_m)

  --  local logpredDA = decode(predictions):reshape(opt.mini_batch_size*opt.max_n*opt.max_m,1):sub(1,opt.max_n)
  local predDA=costToProb(-logpredDA)

  --  print(predDA)
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
    if opt.solution == 'integer' then
      sol = hun:reshape(opt.max_n,1)
    elseif opt.solution == 'distribution' then
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
  --  print(sol)
  sol=getOneHotLab(sol, true, opt.max_m)
  -- plot prob distributions
  
--  print(winTitle)
  if winTitle ~= '' then
    local plotTab = {}
    gnuplot.raw('unset ytics')
    plotTab = getPredPlotTab(plotTab, predDA, 1)
    --  print(sol)
    plotTab = getPredPlotTab(plotTab, sol, 2)
    plot(plotTab, winID, winTitle)
    gnuplot.raw('set ytics')
  end


  return mm
end


function minValAndIt(values)
  local mv, mi = torch.min(values,1)
  mv=mv:squeeze()
  mi=mi:squeeze()*opt.eval_val_every
  return mv, mi
end

function maxValAndIt(values)
  local mv, mi = minValAndIt(-values)
  mv = -mv
  return mv, mi
end
