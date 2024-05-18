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

        self.mlahead_channels = 128
        self.head1_1 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.head1_2 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.head1_3 = nn.Sequential(nn.ConvTranspose2d(128, 128, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.head2_1 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(128), nn.ReLU())
        self.head2_2 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(128), nn.ReLU())
        self.head2_3 = nn.Sequential(nn.ConvTranspose2d(128, 128, 16, stride=8, padding=4, bias=False),
                                     nn.BatchNorm2d(128), nn.ReLU())

        self.down1 = nn.Sequential(nn.Conv2d(2 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(2 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
                                   nn.BatchNorm2d(self.mlahead_channels), nn.ReLU())
        self.down3 = nn.Sequential(nn.Conv2d(2 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
                                   nn.BatchNorm2d(self.mlahead_channels), nn.ReLU())
        # self.head2 = nn.Sequential(nn.ConvTranspose2d(64, 128, 8, stride=4, padding=2,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        # self.head3 = nn.Sequential(nn.ConvTranspose2d(128, 128, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        # self.head4 = nn.Sequential(nn.ConvTranspose2d(320, 128, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(),
        #                            nn.ConvTranspose2d(128, 128, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        # self.head5 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(),
        #                            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),nn.BatchNorm2d(128), nn.ReLU(),
        #                            nn.ConvTranspose2d(128, 128, 16, stride=8, padding=4,bias=False), nn.BatchNorm2d(128), nn.ReLU())


    def forward(self, mla_p2, mla_p3):
        #解码器第一个块
        head1_1 = self.head1_1(mla_p2)
        head2_1 = self.head2_1(mla_p3)
        head1_1 = torch.cat([head1_1,head2_1],dim=1)#torch.Size([4, 256, 14, 14])
        head1_1 = self.down1(head1_1)#torch.Size([4, 128, 14, 14])
        # 解码器第二个块
        head1_2 = self.head1_2(head1_1)
        head2_2 = self.head2_2(head2_1)
        head1_2 = torch.cat([head1_2, head2_2], dim=1)
        head1_2 = self.down2(head1_2)
        # 解码器第三个块
        head1_3 = self.head1_3(head1_2)
        head2_3 = self.head2_3(head2_2)
        head1_3 = torch.cat([head1_3, head2_3], dim=1)
        head1_3 = self.down3(head1_3)


        # head4 = self.head4(mla_p3)
        # # head3=head3+cnnf
        # head3 = self.head3(mla_p4)
        # # head4=head4+cnnf
        # head2 = self.head2(mla_p5)
        # head5=head5+cnnf

        return head1_3#torch.Size([1, 128, 224, 224])



@HEADS.register_module()
class PVT_HeadCnn(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128,
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(PVT_HeadCnn, self).__init__(**kwargs)
        # self.img_size = img_size
        # self.norm_cfg = norm_cfg
        # self.mla_channels = mla_channels
        # self.BatchNorm = norm_layer
        self.mlahead_channels = 128


        self.mlahead = BIMLAHead()
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

    def forward(self, inputs):
        x = self.mlahead(inputs[0], inputs[1]) #x=torch.Size([1, 512（4x128）, 224, 224])

        x = self.global_features(x)#torch.Size([1, 128, 224, 224])
        edge = self.edge(x)#torch.Size([1, 1, 224, 224])
        edge = torch.sigmoid(edge)#torch.Size([1, 1, 224, 224])
        return edge, x
