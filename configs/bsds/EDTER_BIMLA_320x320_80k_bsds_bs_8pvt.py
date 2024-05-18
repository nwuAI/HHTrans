_base_ = [
    '../_base_/models/pvt.py',
    '../_base_/datasets/bsds.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/pvt_large_iter_40000.pth',#训练的时候设置
    # pretrained='../pretrain/pvt_large_iter_40000.pth',#测试的时候设置
    backbone=dict(
        type='PyramidVisionTransformer',
        # style='pytorch'
    ),
    decode_head=dict(
        #由于resnet方法需要使用pvt编码端的信息
        type='PVTResnet_Head'
        # type='PVT_Head'
        #type='PVT_Head_Noise'
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
