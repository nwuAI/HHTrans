_base_ = [
    '../_base_/models/pvt_fpn.py',
    '../_base_/datasets/bsds1.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    type='EncoderDecoder_PvtBifusion',
    pretrained='pretrain/resnet34-333f7ec4.pth',#训练的时候设置
    # pretrained='../pretrain/resnet34-333f7ec4.pth',#测试的时候设置
    backbone=dict(
        type='PvtResNetNoSe'
        # style='pytorch'
    ),
    decode_head=dict(
        type='PVTResnet_Bifusion' # resnet34
    )
)

optimizer = dict(lr=1e-6, weight_decay=0.0002,
paramwise_cfg = dict(custom_keys={'head': dict(lr_mult=10.),'global_model': dict(lr_mult=0.),})
)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-8, by_epoch=False)

test_cfg = dict(mode='slide', crop_size=(224, 224), stride=(200, 200))
find_unused_parameters = True
data = dict(samples_per_gpu=4)
