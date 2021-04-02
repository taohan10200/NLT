from  torchvision import models
import sys
sys.path.append(r"/media/D/ht/C-3-Framework-self")
from  misc.layer import Conv2d,FC
import torch.nn.functional as F
from misc.utils import *
from misc.layer import *
from torchsummary import summary
ResNet_path = '/media/D/ht/PyTorch_Pretrained/resnet50-19c8e357.pth'
VGG_path    = "/media/D/ht/PyTorch_Pretrained/vgg16-397923af.pth"


class VGG_ICASSP(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG_ICASSP, self).__init__()
        vgg = models.vgg16_bn(pretrained=False)
        # if pretrained:
        #     vgg.load_state_dict(torch.load(VGG_path))
        features = list(vgg.features.children())
        print(vgg)
        self.layer1 = nn.Sequential(*features[0:16])
        self.layer2 = nn.Sequential(*features[16:23])
        self.layer3 = nn.Sequential(*features[23:42])


        self.de_pred = nn.Sequential(

            BasicConv(512, 256, kernel_size=1, padding=0, use_bn='bn'),
            # BasicDeconv(128, 128, 2, stride=2, use_bn='bn'),
            nn.Upsample(scale_factor = 2,mode='bilinear'),
            BasicConv(256, 128, kernel_size=9, padding=4,  use_bn='bn'),
            # BasicDeconv(64, 64, 2, stride=2, use_bn='bn'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            BasicConv(128, 64, kernel_size=7, padding=3,  use_bn='bn'),
            # BasicDeconv(32, 32, 2, stride=2, use_bn='bn'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            BasicConv(64, 32, kernel_size=5, padding=2,  use_bn='bn'),
            # BasicDeconv(16, 16, 2, stride=2, use_bn='bn'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            BasicConv(32, 32, kernel_size=3, padding=1,  use_bn='bn'),
            BasicConv(32, 1, kernel_size=1, padding=0)
        )
        self.head = nn.Sequential(PSPModule(512, 256),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
                                  )
        initialize_weights(self.de_pred)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        feature = self.layer3(x)
        pre_map = self.de_pred(feature)
        # pre_mask = self.head(feature)
        # pre_mask = F.interpolate(pre_mask,scale_factor=16,mode='bilinear')
        return feature, pre_map#,pre_mask

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 4, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            # InPlaceABNSync(out_features),
            nn.BatchNorm2d(out_features),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        # bn = InPlaceABNSync(out_features)
        bn = nn.BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        res = models.resnet50()
        pre_wts = torch.load(ResNet_path)
        res.load_state_dict(pre_wts)
        self.frontend = nn.Sequential(res.conv1,res.bn1,res.relu,res.maxpool)
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = make_res_layer(Bottleneck, 256, 6, stride=1)
        self.layer3.load_state_dict(res.layer3.state_dict())

        self.de_pred = nn.Sequential(
                        BasicConv(1024, 128, kernel_size=1, padding=0, use_bn='bn'),
                        BasicConv(128, 64, kernel_size=9, padding=4, use_bn='bn', ),
                        BasicDeconv(64, 64, 2, stride=2, use_bn='bn'),

                        BasicConv(64, 32, kernel_size=7, padding=3, use_bn='bn'),
                        BasicDeconv(32, 32, 2, stride=2, use_bn='bn'),

                        BasicConv(32, 16, kernel_size=5, padding=2, use_bn='bn'),
                        BasicDeconv(16, 16, 2, stride=2, use_bn='bn'),
                        BasicConv(16, 16, kernel_size=3, padding=1, use_bn='bn'),
                        )

        self.atribute = nn.Sequential(
            BasicConv(16, 1, kernel_size=1, padding=0, use_bn='none'),
                                      )
        self.head = nn.Sequential(PSPModule(1024, 512),
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
                                  )
        # self.de_pred = nn.Sequential(Conv2d(1024, 128, 1, same_padding=True, NL='relu'),
        #                              Conv2d(128, 1, 1, same_padding=True, NL='relu'))
        initialize_weights( self.de_pred)
    def forward(self,x):
        x = self.frontend(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        pre_map = self.de_pred(layer3)
        # pre_mask = self.head(layer3)
        # pre_mask = F.interpolate(pre_mask,scale_factor=8,mode='bilinear')
        # pre_map = pre_map*pre_mask
        pre_map = self.atribute(pre_map)

        return layer3, pre_map #,pre_mask

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
    from config import  cfg
    from thop import profile

    model = VGG().cuda()
    print(model)
    summary(model,(3,64 ,64 ),batch_size=4)

    input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(model, inputs=(input,))
    print(flops)
    print(params)
