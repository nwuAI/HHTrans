norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VIT_BIMLA',
        model_name='vit_large_patch16_384',
        img_size=320,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=19,
        drop_rate=0.0,
        norm_cfg=dict(type='BN', requires_grad=True),
        pos_embed_interp=True,
        align_corners=False,
        mla_channels=256,
        mla_index=(5, 11, 17, 23)),
    decode_head=dict(
        type='VIT_BIMLAHead',
        in_channels=1024,
        channels=512,
        img_size=320,
        mla_channels=256,
        mlahead_channels=128,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='HEDLoss', use_sigmoid=True, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='VIT_BIMLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=0,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='VIT_BIMLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=1,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='VIT_BIMLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=2,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='VIT_BIMLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=3,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='VIT_BIMLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=4,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='VIT_BIMLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=5,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='VIT_BIMLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=6,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='VIT_BIMLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=7,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4))
    ])
train_cfg = dict()
test_cfg = dict(mode='slide', crop_size=(320, 320), stride=(280, 280))
dataset_type = 'BSDSDataset'
data_root = '../data/BSDS'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
crop_size = (320, 320)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCropTrain', crop_size=(320, 320), cat_max_ratio=0.75),
    dict(type='PadBSDS', size=(320, 320), pad_val=0, seg_pad_val=255),
    dict(
        type='NormalizeBSDS',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        to_rgb=True),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 320),
        flip=False,
        transforms=[
            dict(
                type='NormalizeBSDSTest',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                to_rgb=True),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='BSDSDataset',
        data_root='../data/BSDS',
        img_dir='',
        ann_dir='',
        split='ImageSets/train_pair.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomCropTrain',
                crop_size=(320, 320),
                cat_max_ratio=0.75),
            dict(type='PadBSDS', size=(320, 320), pad_val=0, seg_pad_val=255),
            dict(
                type='NormalizeBSDS',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                to_rgb=True),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='BSDSDataset',
        data_root='../data/BSDS',
        img_dir='',
        ann_dir='',
        split='ImageSets/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 320),
                flip=False,
                transforms=[
                    dict(
                        type='NormalizeBSDSTest',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        to_rgb=True),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='BSDSDataset',
        data_root='../data/BSDS',
        img_dir='',
        ann_dir='',
        split='ImageSets/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 320),
                flip=False,
                transforms=[
                    dict(
                        type='NormalizeBSDSTest',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        to_rgb=True),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=20, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='SGD',
    lr=1e-06,
    momentum=0.9,
    weight_decay=0.0002,
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-08, by_epoch=False)
total_iters = 80000
checkpoint_config = dict(by_epoch=False, interval=20000)
evaluation = dict(interval=20000, metric='mIoU')
find_unused_parameters = True
work_dir = '../result'
gpu_ids = range(0, 1)
