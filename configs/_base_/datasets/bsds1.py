# dataset settings
dataset_type = 'BSDSDataset'
# data_root = '../HED-BSDS'#训练的时候设置
data_root = '../../HED-BSDS'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
crop_size = (224, 224)#(320,320)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    #dict(type='Resize', img_scale=(2048, 320), ratio_range=(0.5, 2.0)),
    dict(type='RandomCropTrain', crop_size=crop_size, cat_max_ratio=0.75),#在图像范围内，截取一个crop_size大小的区域。cat_max_ratio=0.75，表示标记图像的某一个类别的有效标记区域占整图的比例要小于cat_max_ratio。
    #dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='PadBSDS', size=crop_size, pad_val=0, seg_pad_val=255),
    #dict(type='DefaultFormatBundle'),
    dict(type='NormalizeBSDS', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 224),#320
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            #dict(type='Resize', keep_ratio=True),
            #dict(type='RandomFlip'),
            dict(type='NormalizeBSDSTest', **img_norm_cfg),
            #dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        split='ImageSets/train_pair.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        split='ImageSets/test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        split='ImageSets/test1.txt',
        pipeline=test_pipeline))
