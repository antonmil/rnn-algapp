require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'lfs'
require 'image'
require 'mattorch'
require 'util.misc'

nngraph.setDebug(true)

inp=1;  -- dimensionality of one sequence element
outp=1; -- number of derived features for one sequence element
kw=3;   -- kernel only operates on one sequence element per step
dw=1;   -- we step once and go on to the next sequence element

mlp=nn.TemporalConvolution(inp,outp,kw,dw)
print(mlp:getParameters():nElement())
print(nn.Linear(7,24):getParameters():nElement())

x=torch.rand(7,inp) -- a sequence of 7 elements
print(mlp:forward(x):size())

