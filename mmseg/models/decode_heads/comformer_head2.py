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
    def __init__(self,in_channels=768, mla_channels=256, norm_cfg=None):
        super(BIMLAHead, self).__init__()
        self.mlahead_channels = 128
        self.mla_p5_1x1 = nn.Sequential(nn.Conv2d(in_channels, mla_channels, 1, bias=False),
                                        build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.transhead = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 8, stride=4, padding=2,bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 8, stride=4, padding=2,bias=False), nn.BatchNorm2d(128), nn.ReLU())

        # self.head3 = nn.Sequential(nn.ConvTranspose2d(128, 128, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(128), nn.ReLU())

    def to_2D(self, x):
        n, hw, c = x.shape
        h=w = int(math.sqrt(hw))
        x = x.transpose(1,2).reshape(n, c, h, w)
        return x


    def forward(self, mla_p2):
        res2 = self.to_2D(mla_p2)  # [1, 768, 14, 14]
        mla_p5_1x1 = self.mla_p5_1x1(res2)# [1, 256, 14, 14]
        transhead = self.transhead(mla_p5_1x1)

        return transhead



@HEADS.register_module()
class Conformer_Head2(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128,
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(Conformer_Head2, self).__init__(**kwargs)
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        # self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels


        self.mlahead = BIMLAHead(mla_channels=self.mla_channels, norm_cfg=self.norm_cfg)
        self.global_features = nn.Sequential(

            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU())
        self.edge = nn.Conv2d(self.mlahead_channels, 1, 1)

    def forward(self, inputs):
        x = inputs[1][:, 1:]
        x = self.mlahead(x) #x=torch.Size([1, 512（4x128）, 224, 224])
        x = self.global_features(x)#torch.Size([1, 128, 224, 224])
        edge = self.edge(x)#torch.Size([1, 1, 224, 224])
        edge = torch.sigmoid(edge)#torch.Size([1, 1, 224, 224])
        return edge, x
