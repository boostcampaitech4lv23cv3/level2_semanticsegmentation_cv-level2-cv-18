# dataset settings

# from mmseg.datasets.builder import PIPELINES
# from mmcls.datasets.pipelines.transforms import Albu
# PIPELINES.register_module(module=Albu)

dataset_type = 'CustomTrashDataset'
data_root = '/opt/ml/input/data'
classes = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_size = (512, 512)

albu_train_transforms =[
            dict(
                type='RandomResizedCrop',
                width=512,
                height=512,
                scale=(0.5,0.8),
                p=0.8
            ),
            dict(
                type='OneOf',
                transforms=[
                    #dict(type='Flip', p=1.0),
                    dict(type='RandomRotate90', p=1.0),
                ],
                p=0.9),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur',p=1.0),
                    dict(type='GaussianBlur',p=1.0),
                    dict(type='MedianBlur',p=1.0),
                    dict(type='MotionBlur',p=1.0)
                ],
                p=0.1
    
            ),      
            dict(
                type='ColorJitter',
                brightness=0.2, contrast=0.0, saturation=0.0, hue=0.5),

            dict(type='GaussNoise', var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
            dict(type='ToGray' ,p=0.2)
        ]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CustomLoadAnnotations', coco_json_path = '/opt/ml/input/data/train.json'), #, reduce_zero_label=True),
    dict(type='Resize', img_scale=img_size),
    #dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap={
            'img': 'image',
            'gt_semantic_seg': 'mask',
        },
        update_pad_shape=False,
        ),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # reduce_zero_label=True,
        img_dir=data_root,
        coco_json_path = '/opt/ml/input/data/train.json',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        # reduce_zero_label=True,
        img_dir=data_root,
        coco_json_path = '/opt/ml/input/data/val.json',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # reduce_zero_label=True,
        img_dir=data_root,
        coco_json_path = '/opt/ml/input/data/test.json',
        classes=classes,
        pipeline=test_pipeline))
