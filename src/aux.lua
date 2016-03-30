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

--------------------------------------------------------------------------
--- Generate Data for Hungarian training
function genHunData(nSamples)
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

function findFeasibleSolutions(N,M)
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
  local loccost = probs:clone():reshape(opt.mini_batch_size, opt.max_n*opt.nClasses)
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
-- @param P     The probability matrix
-- @param PredP Predicted (marginal) probabilities matrix
function printDebugValues(P, PredP)

  local C = probToCost(P) -- negative log-probability (cost)
  local N,M = getDataSize(P)
  --  local P = torch.exp(-C)
  --  for i = 1,N do
  --    P[i] = P[i]/torch.sum(P[i])
  --  end


  --  print('Cost matrix')
  local minv, mini = torch.min(C,2)
  minv=minv:reshape(N) mini=mini:reshape(N)

  local HunAss = hungarianL(C)
  local marginals = getMarginals(P,solTable)

  local mmaxv, mmaxi = torch.max(marginals,2)
  mmaxv=mmaxv:reshape(N) mmaxi=mmaxi:reshape(N)

  local pmmaxv, pmmaxi = torch.max(PredP,2)
  pmmaxv=pmmaxv:reshape(N) pmmaxi=pmmaxi:reshape(N)



  print(string.format('%5s%5s%5s%5s%5s%5s|%s|%s|%s','i','NN','HA','Mar','PMar','Err',
    getDebugTableTitle('Prob'),getDebugTableTitle('Marg'),getDebugTableTitle('Predicted Marg')))
  for i=1,N do
    local prLine = ''
    prLine = prLine .. string.format('%5d%5d%5d%5d%5d%5d|',i,mini[i],HunAss[i][2],mmaxi[i],pmmaxi[i],mmaxi[i]-pmmaxi[i])
    for j=1,M do
      --      prLine = prLine ..  string.format('%8.4f',C[i][j])
      prLine = prLine ..  string.format('%8.4f',P[i][j])
    end
    prLine = prLine .. ' |'
    for j=1,M do
      prLine = prLine ..  string.format('%8.4f',marginals[i][j])
    end
    prLine = prLine .. ' |'
    for j=1,M do
      prLine = prLine ..  string.format('%8.4f',PredP[i][j])
    end

    --    prLine=prLine..'\n'
    print(prLine)
  end
  --  print(probMat)
  --  print(mini:long():reshape(N,1))
  local NNcost = torch.prod(P:gather(2,mini:long():reshape(N,1)))
  local HAcost = torch.prod(P:gather(2,HunAss:narrow(2,2,1):long():reshape(N,1)))
  local Marcost = torch.prod(P:gather(2,mmaxi:long():reshape(N,1)))
  local PMarcost = torch.prod(P:gather(2,pmmaxi:long():reshape(N,1)))
  local MMsum = torch.sum(mmaxi-pmmaxi)
  --  print(NNcost,HAcost,Marcost,PMarcost)
  print(string.format('%5s%5.2f%5.2f%5.2f%5.2f%5d','Prod',NNcost,HAcost,Marcost,PMarcost,MMsum))
  --  print(C)
  --  print(mini)
  --  print(C:index(1,mini))
  --  print(string.format('%8s%8.4f%8.4f%8.4f|  ---  Cost ---','-',torch.sum(C[mini]),torch.sum(C[mmaxi]),0))

  --  print('NN')
  --  print(mini)
  --  print('Hungarian')
  --  print(HunAss)
  --  print('Marginals')
  --  print(marginals)
  --  print(mmaxi)


end

function createAuxDirs()
  mkdirP('./data')
  mkdirP('./bin')
  mkdirP('./tmp')
  mkdirP('./out')
  mkdirP('./config')
end
