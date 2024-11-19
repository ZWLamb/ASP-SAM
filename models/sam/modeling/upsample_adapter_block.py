import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn.init import xavier_uniform_


warnings.filterwarnings('ignore')
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

'''
题目：Learning to Upsample by Learning to Sample
即插即用的上采样模块：DySample

我们推出了 DySample，这是一款超轻且高效的动态上采样器。
虽然最近基于内核的动态上采样器（如 CARAFE、FADE 和 SAPA）取得了令人印象深刻的性能提升，
但它们引入了大量工作负载，主要是由于耗时的动态卷积和用于生成动态内核的额外子网络。
我们实现了一个新上采样器 DySample。

该上采样适用于：语义分割、目标检测、实例分割、全景分割。
style='lp' / ‘pl’ 用该模块上采样之前弄清楚这两种风格
'''

class GateGenerator(nn.Module):
    def __init__(self, in_channels):
        super(GateGenerator, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.weights_init_xavier_uniform()

    def forward(self, x):
        return torch.sigmoid(F.interpolate(self.conv(x), scale_factor=2))#2 3 4 4 -> 2 1 4 4

    def weights_init_xavier_uniform(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class AdapterDecoder(nn.Module):
    def __init__(self, in_channels=64, scale=2, groups=4, dyscope=False,in_channels_de=None):
        super(AdapterDecoder,self).__init__()

        in_channels_de = in_channels_de if in_channels_de is not None else in_channels

        self.scale = scale
        self.groups = groups

        assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0


        in_channels = in_channels // scale ** 2
        out_channels = 2 * groups


        self.AdapterOffset = nn.Conv2d(in_channels, int(in_channels/2), 1)
        normal_init(self.AdapterOffset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.AdapterGateGenerator = GateGenerator(in_channels_de)
        self.AdapterPreUpsample = nn.Conv2d(int(in_channels/2), in_channels, 1)
        #self.register_buffer('init_pos', self._init_pos())

        ##
        self.AdapterNorm = nn.BatchNorm2d(int(in_channels))
        self.act = nn.ELU()
        ##
    # def _init_pos(self):
    #     h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
    #     return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, int(self.groups/4), 1).reshape(1, -1, 1, 1)

    def forward_pl(self, x): #1*64*4*4 8*32*64*64
        x_ = F.pixel_shuffle(x, self.scale)  #像素重排 x_: 1*16*8*8  8*8*128*128
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.AdapterOffset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = self.AdapterOffset(x_) * 0.25 #+ self.init_pos  #1*8*8*8

        gate = self.AdapterGateGenerator(x)     #8*1*128*128

        if gate.shape[-1]==256:
            nm = self.AdapterNorm(gate * x_ + (1 - gate) * self.AdapterPreUpsample(offset))
            rl = self.act(nm)
        else:
            rl = gate * x_ + (1 - gate) * self.AdapterPreUpsample(offset)
        return rl

    def forward(self, x):
        return self.forward_pl(x)

    '''
    # 'lp' (局部感知):
    这种风格直接在输入特征图的每个局部区域生成偏移量，然后基于这些偏移进行上采样。
    这意味着每个输出像素的位置都是由其对应输入区域内的内容直接影响的，
    适用于需要精细控制每个输出位置如何从输入特征中取样的情况。
    在需要保持局部特征连续性和细节信息的任务（如图像超分辨率、细节增强）中，'lp' 风格可能会表现得更好。
  
    # 'pl' (像素shuffle后局部感知):
    在应用偏移量之前，首先通过像素shuffle操作打乱输入特征图的像素排列，
    这实质上是一种空间重排，能够促进通道间的信息交互。随后，再进行与'lp'类似的局部感知上采样。
    这种风格可能更有利于全局上下文的融合和特征的重新组织，适合于那些需要较强依赖于相邻区域上下文信息的任务
    （例如语义分割，全景分割）。像素shuffle增加了特征图的表征能力，有助于模型捕捉更广泛的上下文信息。
 
    # 两者各有优势，依赖于特定任务的需求：
    如果任务强调保留和增强局部细节，那么 'lp' 可能是更好的选择。
    如果任务需要更多的全局上下文信息和特征重组，'pl' 可能更合适。
    '''

if __name__ == '__main__':
    # input = torch.rand(8,32,64,64) #8,32,64,64  1，64，4，4
    input = torch.rand(8, 8, 128, 128) #8,32,64,64  1，64，4，4
    AdapterDecoder = AdapterDecoder(in_channels=8,scale=2)
    output = AdapterDecoder(input)
    print('input_size:', input.size())
    print('output_size:', output.size())