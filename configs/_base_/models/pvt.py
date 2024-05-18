# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    # pretrained1='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        # pretrained=None
        # depth=50,
        #num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # dilations=(1, 1, 1, 1),
        # strides=(1, 2, 2, 2),
        # norm_cfg=norm_cfg,
        # norm_eval=False,
        # style='pytorch',
        # contract_dilation=True
        ),

    decode_head=dict(
        type='PVT_Head',
        in_channels=1024,
        channels=512,
        # in_channels=[256, 256, 256, 256],
        # in_index=[0, 1, 2, 3],
        # feature_strides=[4, 8, 16, 32],
        # channels=128,
        # dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=False, loss_weight=1.0)))
# model training and testing settings
train_cfg=dict()
test_cfg=dict(mode='whole')
