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

mlp = nn.DotProduct()
x = torch.rand(5,1,4)
y = torch.rand(5,4,1)
print(x)
print(y)
print(mlp:forward({x, y}))