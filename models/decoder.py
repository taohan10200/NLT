import torch.nn as nn
import torch
import torch.nn.functional as F
from models.conv2d_nlt import Conv2d_nlt
from models.batchnorm import nlt_BatchNorm2d
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride=1,padding=0 ,use_bn=False,nlt=False):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn

        if nlt:
            self.conv = Conv2d_nlt(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                  padding=padding)
            self.bn = nlt_BatchNorm2d(out_channels) if self.use_bn else None

        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding)
            self.bn = nn.BatchNorm2d(out_channels) if self.use_bn else None
        # self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None


    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)

class decoder(nn.Module):
    def __init__(self,feature_channel=512,nlt=False):
        super(decoder,self).__init__()
        self.de_pred = nn.Sequential(
            BasicConv(feature_channel, 256,kernel_size=1,use_bn=True,nlt=nlt),
            nn.UpsamplingNearest2d(scale_factor=2),

            BasicConv(256, 128, kernel_size=3, padding=1, use_bn=True,nlt=nlt),
            nn.UpsamplingNearest2d(scale_factor=2),

            BasicConv(128, 64, kernel_size=3, padding=1, use_bn=True,nlt=nlt),
            nn.UpsamplingNearest2d(scale_factor=2),

            BasicConv(64, 64, kernel_size=3, padding=1, use_bn=True,nlt=nlt),
            BasicConv(64, 1, kernel_size=1, use_bn=False, padding=0,nlt=nlt)
        )
    def forward(self, x):
        x = self.de_pred(x)
        return x


class upsampler(nn.Module):
    def __init__(self,feature_channel=512,mtl=False):
        super(upsampler,self).__init__()
        self.conv = BasicConv(feature_channel, 1, kernel_size=3, padding=1, use_bn=False, mtl=mtl)
        self.up = nn.UpsamplingNearest2d(scale_factor=8)
    def forward(self, x):
        x = self.conv(x)
        x =self.up(x)
        return x

if __name__== '__main__':
    from  torchsummary import summary
    model = decoder().cuda()
    summary(model,(512,68,120), batch_size=5)