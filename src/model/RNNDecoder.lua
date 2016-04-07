--[[
  A RNN Decoder for Pointer Network

]]--

require 'model.ticks'

local RNN = {}

function RNN.rnn(opt)

  local RNNMODE, LSTMMODE, GRUMODE = getRNNMode(opt)

  -- set default parameter values
  local rnnSize = opt.rnn_size
  local dropout = opt.dropout or 0
  local batchMode = opt.mini_batch_size>1




  -- input for pairwise distances
  local inputs = {}
  table.insert(inputs, nn.Identity()():annotate{
    name='encoded cost',
    description='Embedded Cost matrix'})


  for L = 1,opt.num_layers do
    -- one input for previous h
    if LSTMMODE then table.insert(inputs, nn.Identity()():annotate{name='c^'..(L)..'_t'}) end
    table.insert(inputs, nn.Identity()():annotate{name='h^'..(L)..'_t'})
  end

--  -- input ecoders weights W1  
--  table.insert(inputs, nn.Identity()())
  
  local eOffset = #inputs  
  -- input encoders hidden states
--  for t=1,opt.TE do
  table.insert(inputs, nn.Identity()())
--  end
  
    
  


--  local inSize = opt.max_n * opt.nClasses
  local inSize = 1 -- input feature vector size

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
--    print(opt.nHiddenInputs*L + 1)
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

--  local da = nn.Linear(opt.rnn_size, opt.nClasses)(top_DA_state):annotate{
--    name='DA_t',
--    description='data assoc.',
--    graphAttributes = {color='green', style='filled'}
--  }
--  local localDaRes = nn.Reshape(opt.nClasses, batchMode)(da):annotate{name='Rshp DA'}
--  local daFinal = localDaRes
--
--  daFinal = nn.LogSoftMax()(localDaRes):annotate{
--    name='DA_t',
--    description='data assoc. LogSoftMax',
--    graphAttributes = {color='green'}
--  }

  --- construct output out of encoder inputs  
  local W2di = nn.Linear(opt.rnn_size, opt.rnn_size)(top_DA_state) -- W2*di
  allEs = inputs[eOffset+1]
--  print(allEs)
--  local oneS =nn.SelectTable(1)(allEs)
--  oneS =nn.SelectTable(1)(oneS)
  print(oneS)
  local stacked = {}
  for t=1,opt.TE do
    if t%(opt.max_m+1)~= 0 then
--    print(eOffset+t)
      local e = nn.SelectTable(t)(allEs) -- select one entry
      e = nn.SelectTable(#DA_state)(e) -- select top most hidden state
      local W1ej = nn.Linear(opt.rnn_size, opt.rnn_size)(e)
      local W1W2sum = nn.CAddTable(){W1ej, W2di}
      local act = nn.Tanh()(W1W2sum)
      local dot = nn.Linear(opt.rnn_size, 1)(act)
       
      table.insert(stacked, dot)
    end
  end
  
  local ui = nn.JoinTable(1,2)(stacked)
  
--  local pCi = nn.Linear(opt.rnn_size,opt.nClasses)(oneS)
  local pCi = nn.LogSoftMax()(ui)


  -- insert hidden states to output
  for _,v in pairs(DA_state) do table.insert(outputs, v) end


  table.insert(outputs,pCi)

  return nn.gModule(inputs, outputs)

end
return RNN
