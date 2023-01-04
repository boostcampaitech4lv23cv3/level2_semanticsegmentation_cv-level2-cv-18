_base_ = [
    '../_base_/models/pointrend_swin.py', '../_base_/datasets/custom_trash.py',
    '../_base_/wandb_runtime.py', '../_base_/schedules/schedule_epoch.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'  # noqa

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=224,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        num_outs=4),
    decode_head=[
    dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=-1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=0.75),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=0.25),
        ]),
    dict(
        type='PointHead',
        in_channels=[256],
        in_index=[0],
        channels=256,
        num_fcs=3,
        coarse_pred_each_layer=True,
        dropout_ratio=-1,
        num_classes=11,
        align_corners=False,
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=0.75),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=0.25),
        ])
])
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=8)
