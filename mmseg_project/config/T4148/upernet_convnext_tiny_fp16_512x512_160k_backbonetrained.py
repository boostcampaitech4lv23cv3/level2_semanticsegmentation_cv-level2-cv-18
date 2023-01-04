_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/custom_trash_albu.py',
    '../_base_/wandb_runtime.py', '../_base_/schedules/T4148_schedule_40k.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
            prefix='backbone.')),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
dataset_type = 'CustomTrashDataset'
data_root = '/opt/ml/input/data'
classes = [
    'Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
    'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_size = (512, 512)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0,
        rotate_limit=30,
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ElasticTransform', p=1.0),
            dict(type='Perspective', p=1.0),
            dict(type='PiecewiseAffine', p=1.0)
        ],
        p=0.3),
    dict(type='Affine', p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                always_apply=False,
                p=1.0),
            dict(type='ChannelShuffle', p=1.0)
        ],
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=0.1,
        contrast_limit=0.15,
        p=0.5),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=15,
        sat_shift_limit=25,
        val_shift_limit=10,
        p=0.5),
    dict(type='GaussNoise', p=0.3),
    dict(type='CLAHE', p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', p=1.0),
            dict(type='GaussianBlur', p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0)
        ],
        p=0.3)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CustomLoadAnnotations',
        coco_json_path='/opt/ml/input/data/train.json'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0,
                rotate_limit=30,
                p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='ElasticTransform', p=1.0),
                    dict(type='Perspective', p=1.0),
                    dict(type='PiecewiseAffine', p=1.0)
                ],
                p=0.3),
            dict(type='Affine', p=0.3),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RGBShift',
                        r_shift_limit=20,
                        g_shift_limit=20,
                        b_shift_limit=20,
                        always_apply=False,
                        p=1.0),
                    dict(type='ChannelShuffle', p=1.0)
                ],
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.1,
                contrast_limit=0.15,
                p=0.5),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=10,
                p=0.5),
            dict(type='GaussNoise', p=0.3),
            dict(type='CLAHE', p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', p=1.0),
                    dict(type='GaussianBlur', p=1.0),
                    dict(type='MedianBlur', blur_limit=5, p=1.0)
                ],
                p=0.3)
        ],
        keymap=dict(img='image', gt_semantic_seg='mask'),
        update_pad_shape=False),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='CustomTrashDataset',
        data_root='/opt/ml/input/data',
        img_dir='/opt/ml/input/data',
        coco_json_path='/opt/ml/input/data/train.json',
        classes=[
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='CustomLoadAnnotations',
                coco_json_path='/opt/ml/input/data/train.json'),
            dict(type='Resize', img_scale=(512, 512)),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.0625,
                        scale_limit=0,
                        rotate_limit=30,
                        p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='ElasticTransform', p=1.0),
                            dict(type='Perspective', p=1.0),
                            dict(type='PiecewiseAffine', p=1.0)
                        ],
                        p=0.3),
                    dict(type='Affine', p=0.3),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RGBShift',
                                r_shift_limit=20,
                                g_shift_limit=20,
                                b_shift_limit=20,
                                always_apply=False,
                                p=1.0),
                            dict(type='ChannelShuffle', p=1.0)
                        ],
                        p=0.5),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=0.1,
                        contrast_limit=0.15,
                        p=0.5),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=15,
                        sat_shift_limit=25,
                        val_shift_limit=10,
                        p=0.5),
                    dict(type='GaussNoise', p=0.3),
                    dict(type='CLAHE', p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', p=1.0),
                            dict(type='GaussianBlur', p=1.0),
                            dict(type='MedianBlur', blur_limit=5, p=1.0)
                        ],
                        p=0.3)
                ],
                keymap=dict(img='image', gt_semantic_seg='mask'),
                update_pad_shape=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CustomTrashDataset',
        data_root='/opt/ml/input/data',
        img_dir='/opt/ml/input/data',
        coco_json_path='/opt/ml/input/data/val.json',
        classes=[
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CustomTrashDataset',
        data_root='/opt/ml/input/data',
        img_dir='/opt/ml/input/data',
        coco_json_path='/opt/ml/input/data/test.json',
        classes=[
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(
            type='MMSegWandbHook',
            init_kwargs=dict(
                entity='light-observer',
                project='Trash_MMseg',
                name='uper_conv_tiny_fullpreTrain'),
            num_eval_images=0)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
lr = 0.0001
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=6))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=22)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')
crop_size = (512, 512)
fp16 = dict()
work_dir = './work_dirs/upernet_convnext_tiny_fp16_512x512_160k_ade20k'
gpu_ids = [0]
auto_resume = False
