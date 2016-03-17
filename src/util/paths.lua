--------------------------------------------------------------------------
--- Get directory where sequences are stored.
function getDataDir()
  -- 'SET PATH TO DATA HERE'
  local dataDir = './data/'
  if lfs.attributes('/media/sf_vmex','mode') then 			-- virtual machine
    dataDir = '/media/sf_vmex/2DMOT2015/data/'
  elseif lfs.attributes('/home/amilan/research/projects/bmtt-data/','mode') then -- PC in office Adelaide
    dataDir = '/home/amilan/research/projects/bmtt-data/'
--     dataDir = '/home/amilan/storage/databases/2DMOT2015/train/'
  elseif lfs.attributes('/home/h3/','mode') then 			-- network
    dataDir = '/home/h3/amilan/storage/databases/2DMOT2015/train/'
  end
  
  -- make sure directory exists
  assert(lfs.attributes(dataDir, 'mode'), 'No data dir found')
  
  return dataDir  
    
end

--------------------------------------------------------------------------
--- Get directory where results are stored.
function getResDir(modelName, modelSign)
  local outDir = 'out'..'/'
 
  if modelName ~= nil and modelSign ~= nil then
    outDir = 'out'..'/'..modelName..'_'..modelSign..'/'
  end
  
  print('Getting results from '..outDir)
  return outDir
end