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

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)



    def forward(self, x1, x2, x3):

        x1_1 = x1#表示最高水平的特征torch.Size([4, 32, 7, 7])
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2#表示X4经过上采样，经过卷积和X3相乘torch.Size([4, 32, 14, 14])
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3#表示X4经过两个上采样，经过卷积层；X3经过一个上采样，经过一个卷积；X3；三者相乘torch.Size([4, 32, 28, 28])


        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)#表示X4和X3最后的特征进行concattorch.Size([4, 64, 14, 14])
        x2_2 = self.conv_concat2(x2_2)#X4和X3concat后经过卷积操torch.Size([4, 64, 14, 14])

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)#将三个尺度的特征融合后与一二尺度的特征进行concattorch.Size([4, 96, 28, 28])
        x3_2 = self.conv_concat3(x3_2)#torch.Size([4, 96, 28, 28])

        x1 = self.conv4(x3_2)#最后的一个卷积torch.Size([4, 32, 28, 28])

        return x1

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class SAM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BIMLAHead(nn.Module):
    def __init__(self):
        super(BIMLAHead, self).__init__()

        self.head234=nn.Sequential(nn.ConvTranspose2d(32, 32, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(32), nn.ReLU())
        #self.head1=nn.Sequential(nn.ConvTranspose2d(64, 32, 8, stride=4, padding=2,bias=False), nn.BatchNorm2d(32), nn.ReLU())


    # def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
    def forward(self, mla_p2):

        head234=self.head234(mla_p2)

        return head234


@HEADS.register_module()
class PVT_Head_Seg(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128,
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(PVT_Head_Seg, self).__init__(**kwargs)
        # self.img_size = img_size
        # self.norm_cfg = norm_cfg
        # self.mla_channels = mla_channels
        # self.BatchNorm = norm_layer
        # self.mlahead_channels = 64
        self.mlahead_channels = 32

        self.Translayer2_0 = BasicConv2d(64, 32, 1)
        self.Translayer2_1 = BasicConv2d(128, 32, 1)
        self.Translayer3_1 = BasicConv2d(320, 32, 1)
        self.Translayer4_1 = BasicConv2d(512, 32, 1)
        self.CFM =CFM(32)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.SAM = SAM()

        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(32, 1, 1)
        self.out_CFM = nn.Conv2d(32, 1, 1)
        # self.mlahead = BIMLAHead()

        # CFM
        # self.scale_att = scale_atten_convblock(256, 4)

        self.global_features = nn.Sequential(
            # nn.Conv2d(4 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.Conv2d(1 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU())
        self.edge = nn.Conv2d(self.mlahead_channels, 1, 1)

    def forward(self, inputs):


        # x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3]) #x=torch.Size([1, 256（4x64）, 224, 224])

        # CIM
        x1 = self.ca(inputs[0]) * inputs[0]  # channel attention
        cim_feature = self.sa(x1) * x1  # spatial attention
        # CFM
        x2_t = self.Translayer2_1(inputs[1])#torch.Size([4, 32, 28, 28])
        x3_t = self.Translayer3_1(inputs[2])#torch.Size([4, 32, 14, 14])
        x4_t = self.Translayer4_1(inputs[3])#torch.Size([4, 32, 7, 7])
        cfm_feature = self.CFM(x4_t, x3_t, x2_t)#torch.Size([4, 32, 28, 28])

        # SAM
        T2 = self.Translayer2_0(cim_feature)
        T2 = self.down05(T2)
        sam_feature = self.SAM(cfm_feature, T2)

        #x=self.mlahead(cfm_feature,inputs[0])
        x=self.mlahead(cfm_feature)

        # x = self.scale_att(x)

        x = self.global_features(x)#torch.Size([1, 64, 224, 224])
        edge = self.edge(x)#torch.Size([1, 1, 224, 224])
        edge = torch.sigmoid(edge)#torch.Size([1, 1, 224, 224])
        return edge, x
