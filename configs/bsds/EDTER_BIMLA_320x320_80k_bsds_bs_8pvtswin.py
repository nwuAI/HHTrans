_base_ = [
    '../_base_/models/swin_pvt.py',
    '../_base_/datasets/bsds1.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    type='SwinEncoderDecoder',
    pretrained='../pretrain/pvt_large_iter_40000.pth',#训练的时候设置
    # pretrained='pretrain/pvt_large_iter_40000.pth',#训练的时候设置
    pretrained1='../pretrain/swin_tiny_patch4_window7_224_22k.pth',#训练的时候设置
    # pretrained1='pretrain/swin_tiny_patch4_window7_224_22k.pth',#训练的时候设置

    backbone=dict(
        type='SwinTransformer',
        # style='pytorch'
    ),
    swindecode_head=dict(
        type='Swin_Head'
    ),
    pvtdecode_head=dict(
        type='PVTSwin_Head'
    )


    # decode_head=dict(num_classes=150)
)

optimizer = dict(lr=1e-6, weight_decay=0.0002,
paramwise_cfg = dict(custom_keys={'head': dict(lr_mult=10.)})
)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-8, by_epoch=False)

test_cfg = dict(mode='slide', crop_size=(224, 224), stride=(200, 200))
find_unused_parameters = True
data = dict(samples_per_gpu=4)
