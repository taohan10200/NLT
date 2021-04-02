import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair

class _ConvNd_nlt(Module):
    """The class for meta-transfer convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd_nlt, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.nlt_weight = Parameter(torch.ones(in_channels, out_channels // groups, 1, 1))
            self.nlt_bias = Parameter(torch.zeros(in_channels, out_channels // groups, 1, 1))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.nlt_weight = Parameter(torch.ones(out_channels, in_channels // groups, 1, 1))
            self.nlt_bias = Parameter(torch.zeros(out_channels,  in_channels // groups, 1, 1))

        self.weight.requires_grad=False
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias.requires_grad=False
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.nlt_weight.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.nlt_bias.data.uniform_(0, 0)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2d_nlt(_ConvNd_nlt):
    """The class for meta-transfer convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_nlt, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, inp):
        new_nlt_weight = self.nlt_weight.expand(self.weight.shape)
        new_nlt_bias = self.nlt_bias.expand(self.weight.shape)
        new_weight = self.weight.mul(new_nlt_weight) +new_nlt_bias

        if self.bias is not None:
            new_bias = self.bias
        else:
            new_bias = None

        return F.conv2d(inp, new_weight, new_bias, self.stride,
                        self.padding, self.dilation, self.groups)
