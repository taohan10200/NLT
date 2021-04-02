from  torchvision import models
from misc.utils import *
from misc.layer import *

class VGG_DACC(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG_DACC, self).__init__()

        vgg = models.vgg16(pretrained=False)
        features = list(vgg.features.children())

        self.layer1 = nn.Sequential(*features[0:16])
        self.layer2 = nn.Sequential(*features[16:23])

        self.de_pred = nn.Sequential(
            BasicConv(512, 128, kernel_size=1, padding=0, use_bn='bn'),
            BasicDeconv(128, 128, 2, stride=2, use_bn='bn'),

            BasicConv(128, 64, kernel_size=3, padding=1,  use_bn='bn'),
            BasicDeconv(64, 64, 2, stride=2, use_bn='bn'),

            BasicConv(64, 32, kernel_size=3, padding=1,  use_bn='bn'),
            BasicDeconv(32, 32, 2, stride=2, use_bn='bn'),

            BasicConv(32, 1, kernel_size=1, padding=0,  use_bn='none')
        )

        initialize_weights(self.de_pred)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.de_pred(x)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        res = models.resnet50(pretrained=True)
        self.frontend = nn.Sequential(res.conv1,res.bn1,res.relu,res.maxpool)
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = make_res_layer(Bottleneck, 256, 6, stride=1)
        self.layer3.load_state_dict(res.layer3.state_dict())

        self.de_pred = nn.Sequential(
                        BasicConv(1024, 128, kernel_size=1, padding=0, use_bn='bn'),
                        BasicConv(128, 64, kernel_size=3, padding=1, use_bn='bn', ),
                        BasicDeconv(64, 64, 2, stride=2, use_bn='bn'),

                        BasicConv(64, 32, kernel_size=3, padding=1, use_bn='bn'),
                        BasicDeconv(32, 32, 2, stride=2, use_bn='bn'),

                        BasicConv(32, 16, kernel_size=3, padding=1, use_bn='bn'),
                        BasicDeconv(16, 16, 2, stride=2, use_bn='bn'),

                        BasicConv(16, 16, kernel_size=3, padding=1, use_bn='bn'),
                        BasicConv(16, 1, kernel_size=1, padding=0, use_bn='none'),

                        )
        initialize_weights( self.de_pred)
    def forward(self,x):
        x = self.frontend(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        pred_map = self.de_pred(layer3)
        return  pred_map

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels,padding, kernel_size,stride=1,use_bn='none'):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn
        if self.use_bn == 'bn':
            self.bn = nn.BatchNorm2d(out_channels)
        elif self.use_bn == 'in':
            self.bn = nn.InstanceNorm2d(out_channels)
        elif self.use_bn == 'none':
            self.bn = None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride,
                              padding=padding,bias=not self.bn)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x, inplace=True)

class BasicDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bn='none'):
        super(BasicDeconv, self).__init__()
        self.use_bn = use_bn
        if self.use_bn == 'bn':
            self.bn = nn.BatchNorm2d(out_channels)
        elif self.use_bn == 'in':
            self.bn = nn.InstanceNorm2d(out_channels)
        elif self.use_bn == 'none':
            self.bn = None
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=0,bias=not self.bn)

    def forward(self, x):
        # pdb.set_trace()
        x = self.tconv(x)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x, inplace=True)

if __name__ == "__main__":

    from  torchsummary import summary
    from thop import profile
    import torch
    from torchvision.models import  vgg16 as vgg_16
    # from resnet_18 import Resnet_18, resnet18
    model = VGG().cuda()
    # print(model)
    summary(model,(3,80,80))

    input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(model, inputs=(input,))
    print(flops)
    print(params)