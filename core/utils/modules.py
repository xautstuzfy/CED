
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, \
    Sequential, \
    Linear, \
    LayerNorm, \
    Conv2d, \
    BatchNorm2d, \
    ReLU, \
    GELU, \
    Identity, AdaptiveAvgPool2d, MaxPool2d
from .stochastic_depth import DropPath


class ConvDownsample(Module):
    def __init__(self, embedding_dim_in, embedding_dim_out):
        super().__init__()
        self.downsample = Conv2d(embedding_dim_in, embedding_dim_out, kernel_size=(3, 3), stride=(2, 2),
                                 padding=(1, 1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        return x.permute(0, 2, 3, 1)



class Decom(nn.Module):
    def __init__(self,
                 embedding_dim_in=128,
                 ):
        super(Decom, self).__init__()

        self.decomconv_d1 = Conv2d(embedding_dim_in, embedding_dim_in, kernel_size=1, dilation=1)
        self.decomconv_d3 = Conv2d(embedding_dim_in, embedding_dim_in, kernel_size=3, padding=2,dilation=2)

    def forward(self, x):
        ones = torch.ones_like(x)
        h_fre = torch.sigmoid((self.decomconv_d1(x) - self.decomconv_d3(x)))
        l_fre = (ones - h_fre) * x
        h_fre = h_fre * x
        return h_fre, l_fre

class Hfre(Module):
    def __init__(self,
                 embedding_dim_in=128,
                 hidden_dim=64,
                 embedding_dim_out=256):
        super(Hfre, self).__init__()

        self.fuse = Conv2d(2*embedding_dim_in, embedding_dim_in, kernel_size=(1,1))
        self.norm = BatchNorm2d(embedding_dim_in)
        self.gelu = GELU()
        self.sigmoid = nn.Sigmoid()
        # dx
        self.conv_x = Sequential(
            Conv2d(embedding_dim_in, 2 * hidden_dim, kernel_size=(1, 1), padding=(0, 0)),
            Conv2d(2 * hidden_dim, 2 * hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
            Conv2d(2 * hidden_dim, embedding_dim_in, kernel_size=(1, 1)),
            BatchNorm2d(embedding_dim_in),
            ReLU(inplace=True)
            )
        # dy
        self.conv_y = Sequential(
            Conv2d(embedding_dim_in, 2 * hidden_dim, kernel_size=(1, 1), padding=(0, 0)),
            Conv2d(2 * hidden_dim, 2 * hidden_dim, kernel_size=(3, 1), padding=(1, 0)),
            Conv2d(2 * hidden_dim, embedding_dim_in, kernel_size=(1, 1)),
            BatchNorm2d(embedding_dim_in),
            ReLU(inplace=True)
            )

    def forward(self, x):
        h_x = self.conv_x(x)
        h_x = self.conv_x(h_x)
        s_x = x * self.sigmoid(h_x)

        h_y = self.conv_y(x)
        h_y = self.conv_x(h_y)
        s_y = x * self.sigmoid(h_y)
        h_fre = x + self.gelu(self.norm(self.fuse(torch.cat([s_x, s_y], dim=1))))
        return h_fre
        #  64


class Lfre(nn.Module):
    def __init__(self,
                 embedding_dim_in=128,
                 hidden_dim=64,
                 embedding_dim_out=256):
        super(Lfre, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv_block = Sequential(
            Conv2d(embedding_dim_in, 2 * hidden_dim, kernel_size=(1, 1), padding=(0, 0),
                   stride=(1,1)),
            Conv2d(2 * hidden_dim, 2 * hidden_dim, kernel_size=(3, 3), padding=(1, 1),
                   stride=(1,1)),
            Conv2d(2 * hidden_dim, embedding_dim_in, kernel_size=(1, 1), padding=(0, 0),
                   stride=(1,1)),
            BatchNorm2d(embedding_dim_in),
            ReLU(inplace=True)
            )
        self.res_block = Sequential(
            Conv2d(embedding_dim_in, 2 * hidden_dim, kernel_size=(3, 3), padding=(1, 1),
                   stride=(1, 1)),
            Conv2d(2 * hidden_dim, embedding_dim_in, kernel_size=(3, 3), padding=(1, 1),
                   stride=(1, 1)),
            BatchNorm2d(embedding_dim_in),
            ReLU(inplace=True),
        )
    def forward(self, x):

        y = self.res_block(self.conv_block(x))

        l_fre = x * self.sigmoid(self.conv_block(y))

        return l_fre



class ConvFuse(Module):
    def __init__(self,
                 embedding_dim_in=128,
                 hidden_dim=192,
                 embedding_dim_out=256):
        super(ConvFuse, self).__init__()
        self.conv_block = Conv2d(2 * embedding_dim_in, embedding_dim_out, kernel_size=(3,3),
                                 padding=(1,1),stride=(1,1))
        self.pool = AdaptiveAvgPool2d((1,1))
        self.fc = Sequential(
            Linear(embedding_dim_out,2 * embedding_dim_out ),
            Linear(2 * embedding_dim_out , embedding_dim_out)
        )
    def forward(self, x):
        _, C, H, W = x.shape
        t = self.conv_block(x)
        t = self.pool(t)
        _, c, h, w = t.shape
        t = self.fc(t.view(-1, c*h*w))
        t = t.view(-1,c,h,w).expand(-1,C,H,W)
        t = t * x
        return t


####convmlp##### # dropout!


class Mlp(Module):
    def __init__(self,
                 embedding_dim_in,
                 hidden_dim=None,
                 embedding_dim_out=None,
                 activation=GELU):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim_in
        embedding_dim_out = embedding_dim_out or embedding_dim_in
        self.fc1 = Linear(embedding_dim_in, hidden_dim)
        self.act = activation()
        self.fc2 = Linear(hidden_dim, embedding_dim_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))



class ConvMLPStage(Module):
    def __init__(self,
                 embedding_dim,
                 dim_feedforward=512,
                 stochastic_depth_rate=0.1):
        super(ConvMLPStage, self).__init__()
        self.norm1 = LayerNorm(embedding_dim)
        self.channel_mlp1 = Mlp(embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward)
        self.norm2 = LayerNorm(embedding_dim)
        self.connect = Conv2d(embedding_dim,
                              embedding_dim,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1),
                              groups=embedding_dim,
                              bias=False)
        self.connect_norm = LayerNorm(embedding_dim)
        self.channel_mlp2 = Mlp(embedding_dim_in=embedding_dim, hidden_dim=dim_feedforward)
        self.drop_path = DropPath(stochastic_depth_rate) if stochastic_depth_rate > 0 else Identity()

    def forward(self, src):
        src = src + self.drop_path(self.channel_mlp1(self.norm1(src)))
        src = self.connect(self.connect_norm(src).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        src = src + self.drop_path(self.channel_mlp2(self.norm2(src)))
        return src


class BasicStage(Module):
    def __init__(self,
                 num_blocks,
                 embedding_dims,
                 stochastic_depth_rate=0.1,
                 downsample=True):
        super(BasicStage, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_blocks)]
        self.blocks = ModuleList()
        block = ConvMLPStage(embedding_dim=embedding_dims[0],
                             stochastic_depth_rate=dpr[0],
                             )
        self.blocks.append(block)
        self.downsample_mlp = ConvDownsample(embedding_dims[0], embedding_dims[1]) if downsample else Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample_mlp(x)
        return x
