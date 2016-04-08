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

local loaded = mattorch.load('../tmp/sparse.mat')
local allSparseQ=loaded.allSparseQ:t()
local allnnz = loaded.allnnz:t()
local nth=3
local nnz = allnnz[nth][1]
print(loaded)
print(nnz)
print(allSparseQ[nth]:narrow(1,1,nnz*2):reshape(nnz,2))
abort()
mlp = nn.SparseLinear(10,3)
local par = (mlp:getParameters())
-- print(par)
local forw = mlp:forward(sparse)
print(forw)

-- print(type(ass)=='table')