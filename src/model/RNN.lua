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
  local da = nn.Linear(opt.rnn_size, opt.nClasses)(top_DA_state):annotate{
    name='DA_t',
    description='data assoc.',
    graphAttributes = {color='green', style='filled'}
  }


  --   localDaRes = nn.Reshape(opt.max_n, opt.nClasses, batchMode)(da):annotate{name='Rshp DA'}
  local localDaRes = nn.Reshape(opt.nClasses, batchMode)(da):annotate{name='Rshp DA'}

  local daFinal = localDaRes

  daFinal = nn.LogSoftMax()(localDaRes):annotate{
    name='DA_t',
    description='data assoc. LogSoftMax',
    graphAttributes = {color='green'}
  }
--   daFinal = nn.Sigmoid()(localDaRes)



  -- insert hidden states to output
  for _,v in pairs(DA_state) do table.insert(outputs, v) end


  table.insert(outputs,daFinal)
  --   if daLoss then table.insert(outputs,nn.Identity()(daFinal)) end


  return nn.gModule(inputs, outputs)

end
return RNN
