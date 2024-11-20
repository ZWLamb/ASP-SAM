import torch
from models.pvtv2 import pvt_v2_b2

from timm.models import named_apply
from functools import partial
from torch import nn
from .utils import _init_weights,act_layer,channel_shuffle,gcd

class ScalePromptGenNet(nn.Module):
    def __init__(self,pretrain = True):
        super(ScalePromptGenNet, self).__init__()
        self.backbone = pvt_v2_b2()
        path = '/home/lang/work/ASAM/models/pretrained_pth/pvt/pvt_v2_b2.pth'
        if pretrain == True:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        self.Head = MultiScaleHead()
        self.gen_scale = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
    def forward(self,x):
        x = self.backbone(x)
        x = self.Head(x)
        output = self.gen_scale(x[3])
        return [output,x[3]]

class PEB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(PEB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = nn.Sequential(
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
        x = self.up(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x) #to thin
        return x


class AFGB(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(AFGB, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.conv_encoder_feature = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.conv_upsample_feature = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.ff = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.conv_encoder_feature(g)
        x1 = self.conv_upsample_feature(x)
        ff = self.activation(g1 + x1)
        ff = self.ff(ff)

        return x * ff


class MultiScaleHead(nn.Module):
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5],):
        super(MultiScaleHead, self).__init__()
        self.sadb4 = SADB(channels[0],channels[0], stride=1, kernel_sizes=kernel_sizes)
        self.sadb3 = SADB(channels[1], channels[1], stride=1, kernel_sizes=kernel_sizes)
        self.sadb2 = SADB(channels[2], channels[2], stride=1, kernel_sizes=kernel_sizes)
        self.sadb1 = SADB(channels[3], channels[3], stride=1, kernel_sizes=kernel_sizes)

        self.afgb3 = AFGB(F_g=channels[1], F_l=channels[1], F_int=channels[1]//2, kernel_size=3, groups=channels[1]//2)
        self.afgb2 = AFGB(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, kernel_size=3,groups = channels[2] // 2)
        self.afgb1 = AFGB(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2, kernel_size=3,groups=channels[3] // 2)

        self.peb4 = PEB(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=1 )
        self.peb3 = PEB(in_channels=channels[1], out_channels=channels[2], kernel_size=3, stride=1)
        self.peb2 = PEB(in_channels=channels[2], out_channels=channels[3], kernel_size=3, stride=1)


    def forward(self, inputs):
        i1,i2,i3,i4 = inputs

        d4 = self.sadb4(i4)
        d3 = self.peb4(d4)
        x3 = self.afgb3(g=d3, x=i3)
        d3 = d3 + x3

        d3 = self.sadb3(d3)
        d2 = self.peb3(d3)
        x2 = self.afgb2(g=d2, x=i2)
        d2 = d2 + x2

        d2 = self.sadb2(d2)
        d1 = self.peb2(d2)
        x1 = self.afgb1(g=d1, x=i1)
        d1 = d1 + x1

        d1 = self.sadb1(d1)

        return [d4,d3,d2,d1]


class SADB(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_sizes, stride, activation='relu6',use_skip_connection = True, expansion_factor = 2,add = True,dw_parallel=True):
        super(SADB, self).__init__()

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

        x = self.pconv1(input)

        msdc_outs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            msdc_outs.append(dw_out)

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





if __name__ =='__main__':
    model = ScalePromptGenNet()
    input = torch.randn(1,3,256,256)
    output = model(input)

    for i in output:
        print(i.shape)