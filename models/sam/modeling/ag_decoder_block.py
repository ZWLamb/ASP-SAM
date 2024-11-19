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

class AGDecoder(nn.Module):
    def __init__(self, in_channels=64, scale=2, groups=4, dyscope=False,in_channels_de=None):
        super(AGDecoder,self).__init__()

        in_channels_de = in_channels_de if in_channels_de is not None else in_channels

        self.scale = scale
        self.groups = groups

        assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        in_channels = in_channels // scale ** 2
        out_channels = 2 * groups

        self.AdapterLocal = nn.Conv2d(in_channels, int(in_channels/2), 1)
        normal_init(self.AdapterLocal, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.AdapterGateGenerator = GateGenerator(in_channels_de)
        self.AdapterPreUpsample = nn.Conv2d(int(in_channels/2), in_channels, 1)

        self.AdapterNorm = nn.BatchNorm2d(int(in_channels))
        self.act = nn.ELU()

    def forward(self, x):
        x_ = F.pixel_shuffle(x, self.scale)  # x_: 1*16*8*8  8*8*128*128
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.AdapterLocal(x_) * self.scope(x_).sigmoid(),
                                       self.scale) * 0.5 + self.init_pos
        else:
            offset = self.AdapterLocal(x_) * 0.25  # + self.init_pos  #1*8*8*8

        gate = self.AdapterGateGenerator(x)  # 8*1*128*128

        if gate.shape[-1] == 256:
            nm = self.AdapterNorm(gate * x_ + (1 - gate) * self.AdapterPreUpsample(offset))
            rl = self.act(nm)
        else:
            rl = gate * x_ + (1 - gate) * self.AdapterPreUpsample(offset)
        return rl

if __name__ == '__main__':
    # input = torch.rand(8,32,64,64) #8,32,64,64  1，64，4，4
    input = torch.rand(8, 8, 128, 128) #8,32,64,64  1，64，4，4
    AGDecoder = AGDecoder(in_channels=8,scale=2)
    output = AGDecoder(input)
    print('input_size:', input.size())
    print('output_size:', output.size())