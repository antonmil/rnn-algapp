--[[
  A RNN Encoder for Pointer Network

]]--

require 'model.ticks'

local RNN = {}

function RNN.rnn(opt)

  local RNNMODE, LSTMMODE, GRUMODE = getRNNMode(opt)

  local rnnSize = opt.rnn_size
  local dropout = opt.dropout or 0
  local batchMode = opt.mini_batch_size>1
  local featDim = 1 -- read in one number at a time



  -- input for pairwise distances
  local inputs = {}
  table.insert(inputs, nn.Identity()():annotate{
    name='One Cost',
    description='One cost element'})


  for L = 1,opt.num_layers do
    -- one input for previous location
    if LSTMMODE then table.insert(inputs, nn.Identity()():annotate{name='c^'..(L)..'_t'}) end
    table.insert(inputs, nn.Identity()():annotate{name='h^'..(L)..'_t'})
  end



--  local inSize = opt.max_n * opt.nClasses
  local inSize = featDim -- input feature vector size

  local x, prev_d, stateDimL, xl, exi, inputSizeL
  local outputs = {}

  -----------------------
  -- ENCODER  --
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


  -- insert hidden states to output
  for _,v in pairs(DA_state) do table.insert(outputs, v) end

  return nn.gModule(inputs, outputs)

end
return RNN
