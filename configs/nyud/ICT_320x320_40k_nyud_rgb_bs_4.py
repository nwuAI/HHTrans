_base_ = [
    '../_base_/models/epsa_cswin.py',
    '../_base_/datasets/nyud_rgb.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    type='CSwinEncoderDecoder',
    # pretrained='../pretrain/cswin_large_22k_224.pth',#训练的时候设置
    pretrained='pretrain/cswin_large_22k_224.pth',#训练的时候设置

    backbone=dict(
        type='CSWinTransformer',
        # style='pytorch'
    ),
    ictdecode_head=dict(
        type='ICT_Head'
    ),
    epsa_auxiliary_head=
        dict(
            type='EPSA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=0,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=1)),

    cswin_auxiliary_head=
        dict(
        type='CSWIN_AUXIHead',
        in_channels=256,
        channels=512,
        in_index=0,
        img_size=320,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=1))

)


optimizer = dict(lr=1e-6, weight_decay=0.0002,
paramwise_cfg = dict(custom_keys={'head': dict(lr_mult=10.)})
)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-8, by_epoch=False)

test_cfg = dict(mode='slide', crop_size=(224, 224), stride=(200, 200))
find_unused_parameters = True
data = dict(samples_per_gpu=1)
