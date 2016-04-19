--[[
  A RNN ...

]]--

require 'model.ticks'

local RNN = {}

function RNN.rnn(opt)

  local RNNMODE, LSTMMODE, GRUMODE = getRNNMode(opt)

  --   assert(false)
  --   print('aa')
  -- set default parameter values
  local rnnSize = opt.rnn_size

  local dropout = opt.dropout or 0

  local batchMode = opt.mini_batch_size>1




  -- input for pairwise distances
  local inputs = {}
  table.insert(inputs, nn.Identity()():annotate{
    name='Cost',
    description='Cost matrix'})


  for L = 1,opt.num_layers do
    -- one input for previous location
    if LSTMMODE then table.insert(inputs, nn.Identity()():annotate{name='c^'..(L)..'_t'}) end
    table.insert(inputs, nn.Identity()():annotate{name='h^'..(L)..'_t'})
  end




--  local inSize = opt.max_n * opt.nClasses
  local inSize = opt.inSize2 -- input feature vector size

  local x, prev_d, stateDimL, xl, exi, inputSizeL
  local outputs = {}
  local pred_state = {}

  local labToH = {}
  --
  -----------------------
  -- DATA  ASSOCIATION --
  -----------------------
  local DA_state = {}
  -- connect top hidden output to new hidden input

  for L=1, opt.num_layers do
    -- hidden input from previous RNN
    local prev_h = inputs[opt.nHiddenInputs*L + 1]
    local prev_c = {}

    if LSTMMODE then prev_c = inputs[2*L] end

    -- real input
    if L==1 then
      x = inputs[1]
      inputSizeL = inSize
    else
      x = DA_state[(L-1)*opt.nHiddenInputs]
      inputSizeL = rnnSize
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
    end

    -- Do one time tick
    local next_c, next_h = {}, {}
    if RNNMODE then DA_state = RNNTick(DA_state, opt, inputSizeL, x, prev_h, L) end
    if LSTMMODE then DA_state = LSTMTick(DA_state, opt, inputSizeL, x, prev_h, prev_c, L) end

  end -- layers



  -- set up the decoder for data association
  local top_DA_state = DA_state[#DA_state]



  if dropout > 0 then top_DA_state = nn.Dropout(dropout)(top_DA_state) end
  local da = nn.Linear(opt.rnn_size, opt.solSize)(top_DA_state):annotate{name='DA_t'}
  local sigDA = nn.Sigmoid()(da)
  local solMatrix = nn.Reshape(opt.max_n, opt.max_m, batchMode)(sigDA)
  local obj = nil
  if opt.problem == 'linear' then
    obj = nn.DotProduct(){inputs[1], sigDA}
  elseif opt.problem == 'quadratic' then
    local costMat = nn.Reshape(opt.max_n*opt.max_n, opt.max_m*opt.max_m, batchMode)(inputs[1])    
    obj = nn.MM(){costMat, nn.Reshape(opt.solSize, 1, batchMode)(sigDA)}
    obj = nn.DotProduct(){sigDA, nn.Reshape(opt.solSize, batchMode)(obj)}
  end 
  
  local constr1 = nn.Sum(1, 2)(solMatrix)
  local constr2 = nn.Sum(2, 2)(solMatrix)


  --   localDaRes = nn.Reshape(opt.max_n, opt.nClasses, batchMode)(da):annotate{name='Rshp DA'}
--  local localDaRes = nn.Reshape(1, batchMode)(obj):annotate{name='Rshp DA'}

  local daFinal = obj

--   daFinal = nn.LogSoftMax()(localDaRes):annotate{
--     name='DA_t',
--     description='data assoc. LogSoftMax',
--     graphAttributes = {color='green'}
--   }
  
--   daFinal = nn.Sigmoid()(localDaRes)



  -- insert hidden states to output
  for _,v in pairs(DA_state) do table.insert(outputs, v) end


  table.insert(outputs,daFinal)
  table.insert(outputs,sigDA)
  table.insert(outputs,constr1)
  table.insert(outputs,constr2)
  --   if daLoss then table.insert(outputs,nn.Identity()(daFinal)) end


  return nn.gModule(inputs, outputs)

end
return RNN
