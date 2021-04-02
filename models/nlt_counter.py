import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import vgg16
from models.resnet_50 import ResNet50
from models.decoder import decoder,upsampler
from collections import OrderedDict


class NLT_Counter(nn.Module):
    def __init__(self, mode=None, backbone='vgg16'):
        super().__init__()
        self.mode = mode

        if self.mode == 'nlt':
            if backbone == 'vgg16':
                print('backbone is vgg16')
                self.encoder = vgg16(nlt=True)
                self.decoder = decoder(feature_channel=512,nlt=True)

            elif backbone == 'ResNet50':
                print('backbone is ResNet50')
                self.encoder = ResNet50(nlt=True)
                self.encoder.load_state_dict(torch.load('/media/D/ht/PyTorch_Pretrained/resnet50-19c8e357.pth'), strict=False)
                self.decoder = decoder(feature_channel=1024, nlt=True)

        else:
            if backbone == 'vgg16':
                self.encoder = vgg16(pretrained=True, nlt=False)
                self.decoder = decoder(feature_channel=512)

            elif backbone == 'ResNet50':
                self.encoder =  ResNet50(nlt=False)
                self.encoder.load_state_dict(torch.load('/media/D/ht/PyTorch_Pretrained/resnet50-19c8e357.pth'), strict=False)
                self.decoder =  decoder(feature_channel=1024, nlt=False)


    def forward(self, inp):

        return self.decoder(self.encoder(inp))

