
--[[

Torch implementation of squeezenet for food recognition.
--]]



 


require 'image'
require 'nn'
 

 

function conv_bn(inp, outp, stride)
local net= nn.Sequential();
net:add((nn.SpatialConvolution(inp, outp,3,3,stride, stride,  1,1)):noBias() )

net:add(nn.SpatialBatchNormalization(outp))
return net;
end

function conv_dw(inp, outp, stride)
local net = nn.Sequential();
net:add(nn.SpatialDepthWiseConvolution(inp, 1,3,3,stride,stride,  1,1) )
net:add(nn.SpatialBatchNormalization(inp))
net:add(nn.ReLU(true))
net:add(nn.SpatialConvolution(inp, outp,1,1,1,1):noBias())
net:add(nn.SpatialBatchNormalization(outp))
net:add(nn.ReLU(true))
return net;
end
  

function  mobilenet(nClasses, H, W)
local model = nn.Sequential();


        
        model:add(conv_bn(  3,  32, 2));
   
            model:add(conv_dw( 32,  64, 1));
            model:add(conv_dw( 64, 128, 2));
            model:add(conv_dw(128, 128, 1));
            model:add(conv_dw(128, 256, 2));
            model:add(conv_dw(256, 256, 1));
            model:add(conv_dw(256, 512, 2));
            model:add(conv_dw(512, 512, 1));
            model:add(conv_dw(512, 512, 1));
            model:add(conv_dw(512, 512, 1));
            model:add(conv_dw(512, 512, 1));
            model:add(conv_dw(512, 512, 1));
            model:add(conv_dw(512, 1024, 2));   
            model:add(conv_dw(1024, 1024, 1));          
            model:add(nn.SpatialAveragePooling(7,7)); 
            -- automatically infer output size for H, W  
          local out= model:forward(torch.randn(1,3,H,W));
          print(out:size())
            model:add(nn.View(out:size(2)*out:size(3)*out:size(4))); 
            model:add(nn.Linear(1024,nClasses))
            model:add(nn.LogSoftMax())
return model
end


 
