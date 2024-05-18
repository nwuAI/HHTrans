norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder_Pvtcnn2',
    pretrained='../pretrain/densenet121-a639ec97.pth',
    backbone=dict(type='PvtDenseNet'),
    decode_head=dict(
        type='PVTResnet_Head1',
        in_channels=1024,
        channels=512,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='HEDLoss', use_sigmoid=False, loss_weight=1.0)),
    fuse_head=dict(
        type='Local8x8_fuse_head',
        in_channels=128,
        channels=128,
        img_size=160,
        num_classes=1,
        align_corners=False,
        loss_decode=dict(type='HEDLoss', use_sigmoid=True, loss_weight=1.0)))
train_cfg = dict()
test_cfg = dict(mode='slide', crop_size=(224, 224), stride=(200, 200))
dataset_type = 'BSDSDataset'
data_root = '../HED-BSDS'
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='BSDSDataset',
        data_root='../HED-BSDS',
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
        data_root='../HED-BSDS',
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
        data_root='../HED-BSDS',
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
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0), global_model=dict(lr_mult=0.0))))
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-08, by_epoch=False)
total_iters = 100000
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='mIoU')
find_unused_parameters = True
work_dir = './work_dirs2/EDTER_BIMLA_320x320_80k_bsds_bs_8pvt_resnet2'
gpu_ids = range(0, 1)
