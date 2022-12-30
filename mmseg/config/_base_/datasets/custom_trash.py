# dataset settings
dataset_type = 'CustomTrashDataset'
data_root = '/opt/ml/input/data'
classes = ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CustomLoadAnnotations', coco_json_path = '/opt/ml/input/data/train.json', reduce_zero_label=True),
    dict(type='Resize', img_scale=img_size, ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        img_dir=data_root,
        coco_json_path = '/opt/ml/input/data/train.json',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        img_dir=data_root,
        coco_json_path = '/opt/ml/input/data/val.json',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        img_dir=data_root,
        coco_json_path = '/opt/ml/input/data/test.json',
        classes=classes,
        pipeline=test_pipeline))
