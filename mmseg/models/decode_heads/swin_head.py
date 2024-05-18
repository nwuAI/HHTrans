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


class BIMLAHead(nn.Module):
    def __init__(self):
        super(BIMLAHead, self).__init__()

        self.head2 = nn.Sequential(nn.ConvTranspose2d(96, 128, 8, stride=4, padding=2,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.head3 = nn.Sequential(nn.ConvTranspose2d(192, 128, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.head4 = nn.Sequential(nn.ConvTranspose2d(384, 128, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                                   nn.ConvTranspose2d(128, 128, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.head5 = nn.Sequential(nn.ConvTranspose2d(768, 128, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                                   nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),nn.BatchNorm2d(128), nn.ReLU(),
                                   nn.ConvTranspose2d(128, 128, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(128), nn.ReLU())


    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = self.head2(mla_p2)
        head3 = self.head3(mla_p3)
        head4 = self.head4(mla_p4)
        head5 = self.head5(mla_p5)


        return torch.cat([head2, head3, head4, head5], dim=1)#torch.Size([1, 128, 224, 224])



@HEADS.register_module()
class Swin_Head(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128,
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(Swin_Head, self).__init__(**kwargs)
        # self.img_size = img_size
        # self.norm_cfg = norm_cfg
        # self.mla_channels = mla_channels
        # self.BatchNorm = norm_layer
        self.mlahead_channels = 128

        # self.mla_p2_1x1 = nn.Sequential(nn.Conv2d(96, 96, 1, bias=False),
        #                                 build_norm_layer(norm_cfg, 96)[1], nn.ReLU())
        # self.mla_p3_1x1 = nn.Sequential(nn.Conv2d(192, 192, 1, bias=False),
        #                                 build_norm_layer(norm_cfg, 192)[1], nn.ReLU())
        # self.mla_p4_1x1 = nn.Sequential(nn.Conv2d(384, 384, 1, bias=False),
        #                                 build_norm_layer(norm_cfg, 384)[1], nn.ReLU())
        # self.mla_p5_1x1 = nn.Sequential(nn.Conv2d(768, 768, 1, bias=False),
        #                                 build_norm_layer(norm_cfg, 768)[1], nn.ReLU())

        self.mlahead = BIMLAHead()
        self.global_features = nn.Sequential(
            nn.Conv2d(4 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU())
        self.edge = nn.Conv2d(self.mlahead_channels, 1, 1)

    def to_2D(self, x):#目的是将hw变为hxw
        n, hw, c = x.shape # n=4,hw=400,c=1024
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, inputs):#[torch.Size([4, 6400, 96]),torch.Size([4, 1600, 192]),torch.Size([4, 400, 384]),torch.Size([4, 100, 768])]

        res1 = self.to_2D(inputs[0])#[4, 80,80, 96]
        res2 = self.to_2D(inputs[1])#[4, 40,40, 192]
        res3 = self.to_2D(inputs[2])#[4, 20,20, 384]
        res4 = self.to_2D(inputs[3])#[4, 10,10, 768]

        x = self.mlahead(res1,res2,res3,res4) #x=torch.Size([1, 512（4x128）, 224, 224])
        x = self.global_features(x)#torch.Size([1, 128, 224, 224])
        edge = self.edge(x)#torch.Size([1, 1, 224, 224])
        edge = torch.sigmoid(edge)#torch.Size([1, 1, 224, 224])
        # 使用resnet的方法：需要使用到pvt编码器的信息
        # return edge, x, inputs
        return edge, x