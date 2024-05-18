from .ann_head import ANNHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .dnl_head import DNLHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .point_head import PointHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .uper_head import UPerHead
from .vit_up_head import VisionTransformerUpHead
from .vit_bimla_auxi_head import VIT_BIMLA_AUXIHead
from .vit_bimla_auxi_head_local8x8 import VIT_BIMLA_AUXIHead_LOCAL8x8
from .local8x8_fuse_head import Local8x8_fuse_head
from .vit_bimla_head import VIT_BIMLAHead
from .vit_bimla_head_local8x8 import VIT_BIMLAHead_LOCAL8x8
from .pvt_head import PVT_Head
from .pvtresnet_head import PVTResnet_Head
from .pvtresnet_head1 import PVTResnet_Head1
from .pvtresnet_head2 import PVTResnet_Head2
from .pvtresnet_head3 import PVTResnet_Head3
from .pvt_head_cnn import PVT_HeadCnn
from .pvt_head_seg import PVT_Head_Seg
from .pvt_head_noise import PVT_Head_Noise
# from .comformer_head import Conformer_Head
# from .comformer_head2 import Conformer_Head2
from .swin_head import Swin_Head
from .pvtswin_head import PVTSwin_Head
from .pvtresnet_bifusion import PVTResnet_Bifusion
from .bifusion_decoder import Bifusion_decoder
from .epsa_auxi_head import EPSA_AUXIHead
from .cswin_auxi_head import CSWIN_AUXIHead
from .ict_head import ICT_Head

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'PointHead', 'VisionTransformerUpHead', 'VIT_BIMLA_AUXIHead', 
    'VIT_BIMLA_AUXIHead_LOCAL8x8', 'Local8x8_fuse_head',
    'VIT_BIMLAHead', 'VIT_BIMLAHead_LOCAL8x8','PVT_Head','PVT_HeadCnn','PVT_Head_Seg',
    'PVT_Head_Noise','PVTResnet_Head','PVTResnet_Head1','PVTResnet_Head2','PVTResnet_Head3','Swin_Head','PVTSwin_Head',
    'PVTResnet_Bifusion','Bifusion_decoder','EPSA_AUXIHead','CSWIN_AUXIHead','ICT_Head'
]

