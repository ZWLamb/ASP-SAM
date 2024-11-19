import torch
from torch import nn
from models.MobileSAMv2.mobilesamv2.pvtv2 import pvt_v2_b2

from timm.models import named_apply
from functools import partial
import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x) #to thin
        return x


class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class MultiScaleHead(nn.Module):
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5],):
        super(MultiScaleHead, self).__init__()
        self.mscb4 = MSDC(channels[0],channels[0], stride=1, kernel_sizes=kernel_sizes)
        self.eucb4 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=1 )

        self.lgag3 = LGAG(F_g=channels[1], F_l=channels[1], F_int=channels[1]//2, kernel_size=3, groups=channels[1]//2)
        self.mscb3 = MSDC(channels[1], channels[1], stride=1, kernel_sizes=kernel_sizes)
        self.eucb3 = EUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=3, stride=1)

        self.lgag2 = LGAG(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, kernel_size=3,
                          groups=channels[2] // 2)
        self.mscb2 = MSDC(channels[2], channels[2], stride=1, kernel_sizes=kernel_sizes)
        self.eucb2 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=3, stride=1)

        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2, kernel_size=3,
                          groups=channels[3] // 2)
        self.mscb1 = MSDC(channels[3], channels[3], stride=1, kernel_sizes=kernel_sizes)
    def forward(self, inputs):
        i1,i2,i3,i4 = inputs

        d4 = self.mscb4(i4)
        d3 = self.eucb4(d4)#上采
        x3 = self.lgag3(g=d3, x=i3)
        d3 = d3 + x3

        d3 = self.mscb3(d3)
        d2 = self.eucb3(d3)
        x2 = self.lgag2(g=d2, x=i2)
        d2 = d2 + x2

        d2 = self.mscb2(d2)
        d1 = self.eucb2(d2)
        x1 = self.lgag1(g=d1, x=i1)
        d1 = d1 + x1

        d1 = self.mscb1(d1)

        return [d4,d3,d2,d1]


class MSDC(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_sizes, stride, activation='relu6',use_skip_connection = True, expansion_factor = 2,add = True,dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel
        self.add = add
        self.expansion_factor = expansion_factor
        self.use_skip_connection = use_skip_connection

        self.ex_channels = int(self.in_channels * self.expansion_factor)

        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.ex_channels, self.ex_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.ex_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])


        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales

        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, input):
        # Apply the convolution layers in a loop

        x = self.pconv1(input)  # 变厚

        msdc_outs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            msdc_outs.append(dw_out)
            # if self.dw_parallel == False:
            #     x = x + dw_out

        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)

        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_channels))

        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                out = self.conv1x1(x)  #未定义
            return input + out
        else:
            return out



class ScalePromptGenNet(nn.Module):
    def __init__(self,pretrain = True):
        super(ScalePromptGenNet, self).__init__()
        self.backbone = pvt_v2_b2()
        #path = '../../pretrained_pth/pvt/pvt_v2_b2.pth'
        path = '/home/lang/work/ASAM/models/pretrained_pth/pvt/pvt_v2_b2.pth'
        if pretrain == True:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        self.Head = MultiScaleHead()
        self.gen_scale = nn.Sequential(
            # nn.Upsample(scale_factor=4,
            #             mode='bilinear',
            #             align_corners=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
    def forward(self,x):
        x = self.backbone(x)
        #64*64*64  128*32*32 320*16*16  512*8*8

        x = self.Head(x)
        output = self.gen_scale(x[3])
        return [output,x[3]]
if __name__ =='__main__':
    model = ScalePromptGenNet()
    input = torch.randn(1,3,256,256)
    output = model(input)

    for i in output:
        print(i.shape)