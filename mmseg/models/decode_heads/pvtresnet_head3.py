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

#对应resnet101骨干生成的decoder
class BIMLAHead(nn.Module):
    def __init__(self):
        super(BIMLAHead, self).__init__()

        self.head2 = nn.Sequential(nn.ConvTranspose2d(32, 32, 8, stride=4, padding=2,bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.head3 = nn.Sequential(nn.ConvTranspose2d(32, 32, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.head4 = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                   nn.ConvTranspose2d(32, 32, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.head5 = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                   nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1, bias=False),nn.BatchNorm2d(32), nn.ReLU(),
                                   nn.ConvTranspose2d(32, 32, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(32), nn.ReLU())


    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = self.head2(mla_p2)
        head3 = self.head3(mla_p3)
        head4 = self.head4(mla_p4)
        head5 = self.head5(mla_p5)


        return torch.cat([head2, head3, head4, head5], dim=1)#torch.Size([1, 128, 224, 224])



@HEADS.register_module()
class PVTResnet_Head3(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128,
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(PVTResnet_Head3, self).__init__(**kwargs)
        # self.img_size = img_size
        # self.norm_cfg = norm_cfg
        # self.mla_channels = mla_channels
        # self.BatchNorm = norm_layer
        # self.mlahead_channels = 128
        self.mlahead_channels = 128


        self.mlahead = BIMLAHead()
        self.global_features = nn.Sequential(
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU())
        self.edge = nn.Conv2d(self.mlahead_channels, 1, 1)

    def forward(self, inputs):
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3]) #x=torch.Size([1, 512（4x128）, 224, 224])
        x = self.global_features(x)#torch.Size([1, 128, 224, 224])
        edge = self.edge(x)#torch.Size([1, 1, 224, 224])
        edge = torch.sigmoid(edge)#torch.Size([1, 1, 224, 224])

        return edge, x