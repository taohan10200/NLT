import torch.nn as nn
import torch
from torchvision import models
from models.layer import convDU,convLR
from misc.utils import *
import torch.nn.functional as F

import pdb

model_path = '/media/D/Models/PyTorch_Pretrained/vgg16-397923af.pth'

class SFCN(nn.Module):
    def __init__(self, ):
        super(SFCN, self).__init__()
        self.seen = 0
        # self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = []
        
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.convDU = convDU(in_out_channels=64,kernel_size=(1,9))
        self.convLR = convLR(in_out_channels=64,kernel_size=(9,1))


        self.output_layer = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1),
                                          nn.ReLU())


        initialize_weights(self.modules())
        vgg = models.vgg16()
        # vgg.load_state_dict(torch.load(model_path))
        features = list(vgg.features.children())
        self.frontend = nn.Sequential(*features[0:23])


    def forward(self,x):
        # pdb.set_trace()
        fea = self.frontend(x)
        x = self.backend(fea)
        x = self.convDU(x)
        x = self.convLR(x)

        x = self.output_layer(x)
        x = F.upsample(x,scale_factor=8)
        # pdb.set_trace()
        return x, fea

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                