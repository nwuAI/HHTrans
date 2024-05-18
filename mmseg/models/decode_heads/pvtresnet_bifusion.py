import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from .helpers import load_pretrained
from .layers import DropPath, to_2tuple, trunc_normal_

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..backbones.vit import Block

from mmcv.cnn import build_norm_layer

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)
        self.relu = nn.ReLU(inplace=True)
        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in  # 最后一个分支

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in  # 第一个分支
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))


class BIMLAHead(nn.Module):
    def __init__(self):
        super(BIMLAHead, self).__init__()

        self.head2 = nn.Sequential(nn.ConvTranspose2d(64, 128, 8, stride=4, padding=2,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.head3 = nn.Sequential(nn.ConvTranspose2d(128, 128, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.head4 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                                   nn.ConvTranspose2d(128, 128, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.head5 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                                   nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),nn.BatchNorm2d(128), nn.ReLU(),
                                   nn.ConvTranspose2d(128, 128, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(128), nn.ReLU())


    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = self.head2(mla_p2)
        head3 = self.head3(mla_p3)
        head4 = self.head4(mla_p4)
        head5 = self.head5(mla_p5)

        return torch.cat([head2, head3, head4, head5], dim=1)#torch.Size([1, 128, 224, 224])



@HEADS.register_module()
class PVTResnet_Bifusion(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128,
                norm_layer=nn.BatchNorm2d, drop_rate=0.2,norm_cfg=None, **kwargs):
        super(PVTResnet_Bifusion, self).__init__(**kwargs)

        self.a_b = nn.Sequential(
            nn.Conv2d(320, 384, 3, padding=1),
            nn.BatchNorm2d(384), nn.ReLU())
        self.pvt_swin = nn.Sequential(
            nn.Conv2d(2 * 384, 384, 3, padding=1),
            nn.BatchNorm2d(384), nn.ReLU())

        self.up1 = Up(in_ch1=384, out_ch=128)
        self.up2 = Up(128, 64)

        self.up_c = BiFusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c_1_1 = BiFusion_block(ch_1=128, ch_2=128, r_2=2, ch_int=128, ch_out=128, drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=64, ch_2=64, r_2=1, ch_int=64, ch_out=64, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(128, 64, 64, attn=True)
        self.drop = nn.Dropout2d(drop_rate)

        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )


        self.mlahead_channels = 64
        self.deconv = nn.Sequential(nn.ConvTranspose2d(64, 64, 8, stride=4, padding=2,bias=False), nn.BatchNorm2d(64), nn.ReLU())

        # self.mlahead = BIMLAHead()
        self.global_features = nn.Sequential(
            # nn.Conv2d(4 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            # nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU())
        self.edge = nn.Conv2d(self.mlahead_channels, 1, 1)

    def forward(self, inputs,global_features):

        # 先对8个特征进行处理
        # global_features：torch.Size([4, 64, 80, 80])，torch.Size([4, 128, 40, 40])，torch.Size([4, 320, 20, 20])，torch.Size([4, 512, 10, 10])
        #                 torch.Size([4, 96, 80, 80])，torch.Size([4, 192, 40, 40])，torch.Size([4, 384, 20, 20])，torch.Size([4, 768, 10, 10])
        # input：torch.Size([4, 64, 80, 80])，torch.Size([4, 128, 40, 40])，torch.Size([4, 256, 20, 20])，torch.Size([4, 512, 10, 10])

        xa=self.a_b(global_features[2])
        xb=torch.cat([xa,global_features[6]],dim=1)
        x_b=self.pvt_swin(xb)#384

        x_b_1 = self.up1(x_b) #transformer下采样一次
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer下采样第二次
        x_b_2 = self.drop(x_b_2)

        # joint path
        x_c = self.up_c(inputs[2], x_b)# 经过bifusion，得到256通道的特征

        x_c_1_1 = self.up_c_1_1(inputs[1], x_b_1)# 经过bifusion，得到128通道的特征
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)# 256上采样到128

        x_c_2_1 = self.up_c_2_1(inputs[0], x_b_2)  # 经过bifusion，得到64通道的特征
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  # 128上采样到64


        # decoder part
        out = self.deconv(x_c_2)
        out =self.global_features(out)
        map_2=self.edge(out)
        # map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        # map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        # map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)
        edge = torch.sigmoid(map_2)

        # x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3]) #x=torch.Size([1, 512（4x128）, 224, 224])
        # x = self.global_features(x)#torch.Size([1, 128, 224, 224])
        # edge = self.edge(x)#torch.Size([1, 1, 224, 224])
        # edge = torch.sigmoid(edge)#torch.Size([1, 1, 224, 224])

        return edge, x_c_2