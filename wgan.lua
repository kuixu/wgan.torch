require 'torch'
require 'nn'
require 'optim'
require 'math'
require 'hdf5'

opt = {
   dataset = 'lsun',       -- cifar10 / imagenet / lsun / folder /lfw
   dataroot = '',          -- path to dataset
   batchSize = 64,         -- input batch size
   loadSize = 96,
   fineSize = 64,          -- the height / width of the input image to network
   imageSize = 64,
   nz = 100,               -- #  of dim for latent Z vector
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate to train for
   Diters = 5,             -- number of D iters per each G iter
   lrD = 0.00005,          -- initial learning rate for Critic
   lrG = 0.00005,           -- initial learning rate for Generator
   beta1 = 0.5,            -- momentum term of adam
   netG = '',               -- path to netG (to continue training)
   netD = '',               -- path to netD (to continue training)
   clamp_lower = -0.01,    
   clamp_upper = 0.01, 
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 0,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'wgan2',
   noise = 'normal',       -- uniform / normal
   n_extra_layers = 0,     -- Number of extra layers on gen and disc
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader

--local DataLoader = paths.dofile('data/data.lua')
--local data = DataLoader.new(opt.nThreads, opt.dataset, opt)

local lfwHd5 = hdf5.open('datasets/lfw.hdf5', 'r')
local data = lfwHd5:read('lfw'):all()
data:mul(2):add(-1)
lfwHd5:close()

print("Dataset: " .. opt.dataset, " Size: ", data:size())

----------------------------------------------------------------------------
--custom weights initialization called on netG and netD
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0


local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution  --2D
local SpatialFullConvolution = nn.SpatialFullConvolution  --2D

local netG = nn.Sequential()

-- input is Z, going into a convolution
netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

netG:apply(weights_init)

--if opt.netG != '' then -- load checkpoint if needed
   --netG = torch.load(opt.netG))
--end
--print(netG)

---------------------------------------------------------
local netD = nn.Sequential()

-- input is (nc) x 64 x 64
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
--netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)

--if opt.netD != '' then -- load checkpoint if needed
   --netD = torch.load(opt.netD))
--end
--print(netD)

local criterion = nn.BCECriterion()
---------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
--local fixed_noise = torch.Tensor(opt.batchSize, nz, 1, 1).normal(0, 1)

local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

local one = torch.Tensor(64, 1):fill(1)
--print("one size", one:size())
--print("one", one)
local mone = one * -1
--print("mone size", mone:size())
--print("mone", mone)

--------------------------------------------------------------------
--RMSProp
optimStateG = {
   learningRate = opt.lrG,
   --beta1 = opt.beta1,  --Adam
}
optimStateD = {
   learningRate = opt.lrD,
   --beta1 = opt.beta1,  --Adam
}

----------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda(); one = one:cuda();  mone = mone:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG, cudnn)
      cudnn.convert(netD, cudnn)
   end
   netD:cuda();           netG:cuda();           criterion:cuda()
end



local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

---------------------------------------------------------------------
-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   --data_tm:reset(); data_tm:resume()
   --local real = data:getBatch()
   --data_tm:stop()
   --input:copy(real)
   label:fill(real_label)

   local output = netD:forward(input)
   --print ("outputD size:", output:size())
   local errD_real = torch.mean(output)    --log(D(X))*1+log(1-D(X))*(1-1)  -- error f(x) = y
   --print("errD_real Size:", errD_real:size())
   local df_do = one             -- df/dy  f(x) =y
   --local errD_real = criterion:forward(output, label)
   --local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake = netG:forward(noise)
   input:copy(fake)
   label:fill(fake_label)

   local output = netD:forward(input)
   local errD_fake = torch.mean(output)   --log(1-D(G(X))*0+log(1-D(G(X))*(1-0) -- error f(x) = y
   local df_do = mone              -- df/dy  f(x)=1-y
   --local errD_fake = criterion:forward(output, label)
   --print("errD_fake Size:", errD_fake:size())
   --local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   errD = errD_real - errD_fake
   --print("errD :", errD)

   return errD, gradParametersD
end

---------------------------------------------------------------------
-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = torch.mean(output)            -- error f(x) = output y
   local df_do = one            -- df/dy  f(x)=y
   --errG = criterion:forward(output, label)
   --print("errG :", errG)
   --local df_do = criterion:backward(output, label)

   local df_dg = netD:updateGradInput(input, df_do)

   netG:backward(noise, df_dg)
   return errG, gradParametersG
end

----------------------------------------------------------------------
function math.Clamp(val, lower, upper)
    assert(val and lower and upper, "not very useful error message here")
    if lower > upper then lower, upper = upper, lower end -- swap if boundaries supplied the wrong way
    return math.max(lower, math.min(upper, val))
end

-----------------------------------------------------------------
--train

local gen_iterations = 0
local Diter = 0

for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0

   local i = 1
   while i < math.min(data:size()[1], opt.ntrain) do
      tm:reset()

      -- train the discriminator Diters times
      if gen_iterations < 25 or gen_iterations % 500 == 0 then
         Diters = 100
      else
         Diters = opt.Diters
      end


      local j = 0
      while j < Diters and i < math.min(data:size()[1], opt.ntrain) do
         j = j + 1

         -- clamp parameters to a cube
         --print ("parametersD size:", parametersD:size(), "parametersD size 1:", parametersD:size(1))
         --for p = 1, parametersD:size(1) do
            --parametersD[p] = math.Clamp(parametersD[p], opt.clamp_lower, opt.clamp_upper)
         --end

         parametersD:clamp(opt.clamp_lower, opt.clamp_upper)

         -- get minibatch input for real
         data_tm:reset(); data_tm:resume()
      
         for k = 1, opt.batchSize do
            local idx = math.min(i+k-1, data:size()[1])  --math.random(data:size()[1])
            local sample = data[idx]     
            input[k] = sample:clone()
         end
         data_tm:stop()

         i = i + opt.batchSize

         -- (1) Update D network: maximize (D(x)) - (D(G(z)))
         optim.rmsprop(fDx, parametersD, optimStateD)


      end

      -- (2) Update G network: maximize (D(G(z))
      optim.rmsprop(fGx, parametersG, optimStateG)
      gen_iterations = gen_iterations + 1

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          local fake = netG:forward(noise_vis)
          local real = data:getBatch()
          disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real, {win=opt.display_id * 3, title=opt.name})
      end



      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f '):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size()[1], opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end

   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))


end
