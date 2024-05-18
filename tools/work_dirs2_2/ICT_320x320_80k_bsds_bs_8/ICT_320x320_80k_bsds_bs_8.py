norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CSwinEncoderDecoder',
    pretrained='../pretrain/cswin_large_22k_224.pth',
    backbone=dict(type='CSWinTransformer'),
    ictdecode_head=dict(
        type='ICT_Head',
        in_channels=1024,
        channels=512,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='HEDLoss', use_sigmoid=False, loss_weight=1.0)),
    epsa_auxiliary_head=[
        dict(
            type='EPSA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=0,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='EPSA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=1,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='EPSA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=2,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='EPSA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=3,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4))
    ],
    cswin_auxiliary_head=[
        dict(
            type='CSWIN_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=0,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='CSWIN_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=1,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='CSWIN_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=2,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='CSWIN_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=3,
            img_size=320,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4))
    ])
train_cfg = dict()
test_cfg = dict(mode='slide', crop_size=(224, 224), stride=(200, 200))
dataset_type = 'BSDSDataset'
data_root = '../../HED-BSDS'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCropTrain', crop_size=(224, 224), cat_max_ratio=0.75),
    dict(type='PadBSDS', size=(224, 224), pad_val=0, seg_pad_val=255),
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
        img_scale=(2048, 224),
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
        data_root='../../HED-BSDS',
        img_dir='',
        ann_dir='',
        split='ImageSets/train_pair.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomCropTrain',
                crop_size=(224, 224),
                cat_max_ratio=0.75),
            dict(type='PadBSDS', size=(224, 224), pad_val=0, seg_pad_val=255),
            dict(
                type='NormalizeBSDS',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                to_rgb=True),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='BSDSDataset',
        data_root='../../HED-BSDS',
        img_dir='',
        ann_dir='',
        split='ImageSets/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 224),
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
        data_root='../../HED-BSDS',
        img_dir='',
        ann_dir='',
        split='ImageSets/test2.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 224),
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
total_iters = 100000
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='mIoU')
find_unused_parameters = True
work_dir = './work_dirs2_2/ICT_320x320_80k_bsds_bs_8'
gpu_ids = range(0, 1)
